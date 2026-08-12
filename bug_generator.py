"""Module 3: Taxonomy-guided adversarial bug generation."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import config
from api_manager import APIManager
from dataset_loader import BenchmarkCase
from evaluator import evaluate_patches, PatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy prompt templates
# ---------------------------------------------------------------------------

TAXONOMY_PROMPTS: dict[str, str] = {
    "variable_misuse": """\
You are an expert adversarial bug injector. Given the following correct {language} program, \
your task is to inject **exactly one** subtle and deceptive **Variable/Data Misuse** bug.

Examples of this category:
- Swapping two similarly-named variables (e.g., using `left` where `right` should be used)
- Using a wrong variable for initialization or return
- Referencing a stale or incorrect variable in a computation

**Rules:**
1. Inject EXACTLY ONE bug — no more.
2. The bug must be subtle and hard to detect by a human reviewer.
3. The bug must cause at least one existing test case to fail.
4. Do NOT change function signatures, imports, or add/remove lines unnecessarily.
5. Return ONLY the complete modified function inside a ```{language}``` code block.

## Correct Code
```{language}
{code}
```
""",

    "logic_error": """\
You are an expert adversarial bug injector. Given the following correct {language} program, \
your task is to inject **exactly one** subtle and deceptive **Logic/Condition Error** bug.

Examples of this category:
- Off-by-one errors in boundary conditions
- Flipping a boolean condition (e.g., `<` to `<=`, `and` to `or`)
- Incorrect comparison or boundary check

**Rules:**
1. Inject EXACTLY ONE bug — no more.
2. The bug must be subtle and hard to detect by a human reviewer.
3. The bug must cause at least one existing test case to fail.
4. Do NOT change function signatures, imports, or add/remove lines unnecessarily.
5. Return ONLY the complete modified function inside a ```{language}``` code block.

## Correct Code
```{language}
{code}
```
""",

    "loop_flaw": """\
You are an expert adversarial bug injector. Given the following correct {language} program, \
your task is to inject **exactly one** subtle and deceptive **Loop/Iteration Flaw** bug.

Examples of this category:
- Wrong loop bounds causing off-by-one or infinite loops
- Skipping elements (e.g., incrementing index incorrectly)
- Using wrong iteration variable or wrong step size

**Rules:**
1. Inject EXACTLY ONE bug — no more.
2. The bug must be subtle and hard to detect by a human reviewer.
3. The bug must cause at least one existing test case to fail.
4. Do NOT change function signatures, imports, or add/remove lines unnecessarily.
5. Return ONLY the complete modified function inside a ```{language}``` code block.

## Correct Code
```{language}
{code}
```
""",

    "param_error": """\
You are an expert adversarial bug injector. Given the following correct {language} program, \
your task is to inject **exactly one** subtle and deceptive **Function Parameter Error** bug.

Examples of this category:
- Passing arguments in the wrong order to a function call
- Using wrong default values
- Passing a slightly wrong argument (e.g., `n` instead of `n-1`)

**Rules:**
1. Inject EXACTLY ONE bug — no more.
2. The bug must be subtle and hard to detect by a human reviewer.
3. The bug must cause at least one existing test case to fail.
4. Do NOT change function signatures, imports, or add/remove lines unnecessarily.
5. Return ONLY the complete modified function inside a ```{language}``` code block.

## Correct Code
```{language}
{code}
```
""",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GeneratedBug:
    program_name: str
    language: str
    generator_model: str
    category: str
    correct_code: str
    buggy_code: str
    test_code: str
    test_path: str
    description: str = ""
    validated: bool = False
    validation_output: str = ""
    checkout_dir: str = ""
    source_file_rel: str = ""


# ---------------------------------------------------------------------------
# Code extraction helper
# ---------------------------------------------------------------------------

def _extract_code_block(response: str, language: str = "python") -> str:
    """Extract the first code block from an LLM response."""
    patterns = [
        rf"```{language}\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        match = re.search(pat, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    stripped = response.strip()
    if stripped and not stripped.startswith("```"):
        return stripped
    return ""


# ---------------------------------------------------------------------------
# Bug generation + validation
# ---------------------------------------------------------------------------

async def generate_bug(
    api: APIManager,
    case: BenchmarkCase,
    model_key: str,
    category: str,
) -> GeneratedBug | None:
    """Generate a single adversarial bug for a program/model/category combo.

    Returns a GeneratedBug if the bug is valid (fails at least one test),
    or None if generation/validation fails.
    """
    template = TAXONOMY_PROMPTS.get(category)
    if not template:
        logger.error("Unknown taxonomy category: %s", category)
        return None

    prompt = template.format(language=case.language, code=case.correct_code)

    logger.info(
        "Generating %s bug for %s using %s",
        config.TAXONOMY[category], case.name, config.MODELS[model_key].display_name,
    )

    responses = await api.generate(
        model_key=model_key,
        prompt=prompt,
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        n_samples=1,
        system="You are an expert adversarial bug injector for software testing research.",
    )

    if not responses or not responses[0]:
        logger.warning("Empty response from %s for %s/%s", model_key, case.name, category)
        return None

    buggy_code = _extract_code_block(responses[0], case.language)
    if not buggy_code:
        logger.warning("Could not extract code block from %s response", model_key)
        return None

    bug = GeneratedBug(
        program_name=case.name,
        language=case.language,
        generator_model=model_key,
        category=category,
        correct_code=case.correct_code,
        buggy_code=buggy_code,
        test_code=case.test_code,
        test_path=case.test_path,
        description=case.description,
        checkout_dir=case.checkout_dir,
        source_file_rel=case.source_file_rel,
    )

    validated = _validate_bug(bug)
    if validated:
        logger.info("Bug validated for %s/%s/%s", case.name, model_key, category)
        return bug
    else:
        logger.info(
            "Bug validation FAILED for %s/%s/%s — buggy code did not fail tests",
            case.name, model_key, category,
        )
        return None


def _validate_bug(bug: GeneratedBug) -> bool:
    """Validate that the buggy code actually fails at least one test.

    Also verifies that the correct code passes all tests (sanity check).
    """
    if bug.language == "python":
        correct_results = evaluate_patches(
            patches=[bug.correct_code],
            program_name=bug.program_name,
            language="python",
            test_code=bug.test_code,
        )
        if not correct_results or not correct_results[0].passed:
            logger.warning(
                "Correct code does not pass tests for %s — skipping validation",
                bug.program_name,
            )
            bug.validated = False
            return False

        buggy_results = evaluate_patches(
            patches=[bug.buggy_code],
            program_name=bug.program_name,
            language="python",
            test_code=bug.test_code,
        )
        bug_failed = buggy_results and not buggy_results[0].passed
        bug.validated = bug_failed
        if buggy_results:
            bug.validation_output = buggy_results[0].test_output or buggy_results[0].error
        return bug_failed

    elif bug.language == "java":
        buggy_results = evaluate_patches(
            patches=[bug.buggy_code],
            program_name=bug.program_name,
            language="java",
            checkout_dir=bug.checkout_dir,
            source_file_rel=bug.source_file_rel,
        )
        bug_failed = buggy_results and not buggy_results[0].passed
        bug.validated = bug_failed
        if buggy_results:
            bug.validation_output = buggy_results[0].test_output or buggy_results[0].error
        return bug_failed

    logger.warning("Unsupported language for validation: %s", bug.language)
    return False


# ---------------------------------------------------------------------------
# Compound bug generation (human bug + additional LLM fault)
# ---------------------------------------------------------------------------

COMPOUND_PROMPT_TEMPLATE = """\
You are an expert adversarial bug injector. The following {language} program \
ALREADY contains a human-induced bug. Your task is to inject **exactly one additional** \
subtle and deceptive **{category_name}** bug on top of the existing fault.

Examples of this category:
{category_examples}

**Rules:**
1. Keep the existing human-induced bug intact — do NOT fix it.
2. Inject EXACTLY ONE additional bug — no more.
3. The additional bug must be subtle and hard to detect by a human reviewer.
4. Do NOT change function signatures, imports, or add/remove lines unnecessarily.
5. Return ONLY the complete modified function inside a ```{language}``` code block.

## Already-Buggy Code (contains a human-induced fault)
```{language}
{code}
```
"""

CATEGORY_EXAMPLES: dict[str, str] = {
    "variable_misuse": (
        "- Swapping two similarly-named variables\n"
        "- Using a wrong variable for initialization or return\n"
        "- Referencing a stale or incorrect variable in a computation"
    ),
    "logic_error": (
        "- Off-by-one errors in boundary conditions\n"
        "- Flipping a boolean condition (e.g., `<` to `<=`, `and` to `or`)\n"
        "- Incorrect comparison or boundary check"
    ),
    "loop_flaw": (
        "- Wrong loop bounds causing off-by-one or infinite loops\n"
        "- Skipping elements (e.g., incrementing index incorrectly)\n"
        "- Using wrong iteration variable or wrong step size"
    ),
    "param_error": (
        "- Passing arguments in the wrong order to a function call\n"
        "- Using wrong default values\n"
        "- Passing a slightly wrong argument (e.g., `n` instead of `n-1`)"
    ),
}


async def generate_compound_bug(
    api: APIManager,
    case: BenchmarkCase,
    model_key: str,
    category: str,
) -> GeneratedBug | None:
    """Inject an additional LLM fault into already-buggy human code.

    Takes a BenchmarkCase whose `buggy_code` already has a human-induced bug
    and prompts the model to add one more fault from the given taxonomy category.
    Validates that the compound-buggy code still fails the test suite.
    """
    if not case.buggy_code.strip():
        logger.warning("No buggy_code for %s — cannot generate compound bug", case.name)
        return None

    category_name = config.TAXONOMY.get(category, category)
    examples = CATEGORY_EXAMPLES.get(category, "")

    prompt = COMPOUND_PROMPT_TEMPLATE.format(
        language=case.language,
        category_name=category_name,
        category_examples=examples,
        code=case.buggy_code,
    )

    logger.info(
        "Generating compound %s bug for %s using %s",
        category_name, case.name, config.MODELS[model_key].display_name,
    )

    responses = await api.generate(
        model_key=model_key,
        prompt=prompt,
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        n_samples=1,
        system="You are an expert adversarial bug injector for software testing research.",
    )

    if not responses or not responses[0]:
        logger.warning("Empty response from %s for compound %s/%s", model_key, case.name, category)
        return None

    compound_code = _extract_code_block(responses[0], case.language)
    if not compound_code:
        logger.warning("Could not extract code block from %s compound response", model_key)
        return None

    bug = GeneratedBug(
        program_name=case.name,
        language=case.language,
        generator_model=model_key,
        category=category,
        correct_code=case.correct_code,
        buggy_code=compound_code,
        test_code=case.test_code,
        test_path=case.test_path,
        description=case.description,
        checkout_dir=case.checkout_dir,
        source_file_rel=case.source_file_rel,
    )

    validated = _validate_bug(bug)
    if validated:
        logger.info("Compound bug validated for %s/%s/%s", case.name, model_key, category)
        return bug
    else:
        logger.info(
            "Compound bug validation FAILED for %s/%s/%s",
            case.name, model_key, category,
        )
        return None


def save_compound_bug(bug: GeneratedBug) -> str:
    """Save a compound bug to the compound bugs directory."""
    out_dir = os.path.join(
        config.COMPOUND_BUGS_DIR, bug.generator_model, bug.category
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bug.program_name}.json")
    data = {
        "program_name": bug.program_name,
        "language": bug.language,
        "generator_model": bug.generator_model,
        "category": bug.category,
        "correct_code": bug.correct_code,
        "buggy_code": bug.buggy_code,
        "test_code": bug.test_code,
        "description": bug.description,
        "validated": bug.validated,
        "validation_output": bug.validation_output[:1000],
        "compound": True,
    }
    if bug.language == "java":
        data["checkout_dir"] = bug.checkout_dir
        data["source_file_rel"] = bug.source_file_rel

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved compound bug to %s", path)
    return path


def load_all_compound_bugs() -> list[GeneratedBug]:
    """Load all previously generated and validated compound bugs from disk."""
    bugs: list[GeneratedBug] = []
    if not os.path.isdir(config.COMPOUND_BUGS_DIR):
        return bugs
    for model_dir in os.listdir(config.COMPOUND_BUGS_DIR):
        model_path = os.path.join(config.COMPOUND_BUGS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        for cat_dir in os.listdir(model_path):
            cat_path = os.path.join(model_path, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if fname.endswith(".json"):
                    try:
                        bug = load_bug(os.path.join(cat_path, fname))
                        if bug.validated:
                            bugs.append(bug)
                    except Exception as exc:
                        logger.error("Failed loading compound bug %s: %s", fname, exc)
    logger.info("Loaded %d validated compound bugs from disk", len(bugs))
    return bugs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_bug(bug: GeneratedBug) -> str:
    """Save a generated bug to JSON. Returns the output path."""
    out_dir = os.path.join(
        config.BUGS_DIR, bug.generator_model, bug.category
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bug.program_name}.json")
    data = {
        "program_name": bug.program_name,
        "language": bug.language,
        "generator_model": bug.generator_model,
        "category": bug.category,
        "correct_code": bug.correct_code,
        "buggy_code": bug.buggy_code,
        "test_code": bug.test_code,
        "description": bug.description,
        "validated": bug.validated,
        "validation_output": bug.validation_output[:1000],
    }
    if bug.language == "java":
        data["checkout_dir"] = bug.checkout_dir
        data["source_file_rel"] = bug.source_file_rel

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved bug to %s", path)
    return path


def load_bug(path: str) -> GeneratedBug:
    """Load a GeneratedBug from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return GeneratedBug(
        program_name=data["program_name"],
        language=data["language"],
        generator_model=data["generator_model"],
        category=data["category"],
        correct_code=data["correct_code"],
        buggy_code=data["buggy_code"],
        test_code=data.get("test_code", ""),
        test_path=data.get("test_path", ""),
        description=data.get("description", ""),
        validated=data.get("validated", False),
        validation_output=data.get("validation_output", ""),
        checkout_dir=data.get("checkout_dir", ""),
        source_file_rel=data.get("source_file_rel", ""),
    )


def load_all_bugs() -> list[GeneratedBug]:
    """Load all previously generated and validated bugs from disk."""
    bugs: list[GeneratedBug] = []
    if not os.path.isdir(config.BUGS_DIR):
        return bugs
    for model_dir in os.listdir(config.BUGS_DIR):
        model_path = os.path.join(config.BUGS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        for cat_dir in os.listdir(model_path):
            cat_path = os.path.join(model_path, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if fname.endswith(".json"):
                    try:
                        bug = load_bug(os.path.join(cat_path, fname))
                        if bug.validated:
                            bugs.append(bug)
                    except Exception as exc:
                        logger.error("Failed loading bug %s: %s", fname, exc)
    logger.info("Loaded %d validated bugs from disk", len(bugs))
    return bugs
