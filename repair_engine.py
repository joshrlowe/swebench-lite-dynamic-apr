"""Module 4: Cross-model repair engine with single-shot structured prompts."""

from __future__ import annotations

import json
import logging
import os
import re

import config
from api_manager import APIManager
from bug_generator import GeneratedBug
from dataset_loader import BenchmarkCase, _extract_python_imports, _clean_buggy_code_for_prompt
from evaluator import (
    BugEvaluation,
    PatchResult,
    build_evaluation,
    evaluate_patches,
    save_evaluation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured repair prompt
# ---------------------------------------------------------------------------

def build_repair_prompt(bug: GeneratedBug, error_output: str = "") -> str:
    """Build a single-shot structured repair prompt with CoT instruction."""
    lang = bug.language.capitalize()

    buggy_code = _clean_buggy_code_for_prompt(bug.buggy_code, bug.program_name)
    sections = [
        f"## Buggy {lang} Code\n"
        f"The following {lang} function contains a bug. "
        f"It is a {config.TAXONOMY.get(bug.category, 'unknown')} type defect.\n\n"
        f"```{bug.language}\n{buggy_code}\n```",
    ]

    if bug.description:
        sections.append(
            f"## Algorithm Description\n{bug.description}"
        )

    if error_output:
        sections.append(
            f"## Failing Test Output\n"
            f"When the buggy code is executed against the test suite, "
            f"the following errors occur:\n\n```\n{error_output[:2000]}\n```"
        )

    sections.append(
        "## Instructions\n"
        "Think step by step through the following:\n"
        "1. Read the buggy code carefully and identify the defect\n"
        "2. Explain what the bug is and why it causes incorrect behavior\n"
        "3. Reason about what the correct fix should be\n"
        "4. Provide the corrected function\n\n"
        f"Return the corrected code inside a ```{bug.language}``` code block. "
        "Include any necessary import statements at the top. Do not include test cases, "
        "comments, or other code — just the function and its imports."
    )

    return "\n\n".join(sections)


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
# Baseline repair prompt (for human-induced bugs without category info)
# ---------------------------------------------------------------------------

def build_baseline_repair_prompt(
    case: BenchmarkCase, error_output: str = ""
) -> str:
    """Build a repair prompt for an original human-induced bug."""
    lang = case.language.capitalize()

    buggy_code = _clean_buggy_code_for_prompt(case.buggy_code, case.name)
    sections = [
        f"## Buggy {lang} Code\n"
        f"The following {lang} function contains a bug.\n\n"
        f"```{case.language}\n{buggy_code}\n```",
    ]

    if case.description:
        sections.append(f"## Algorithm Description\n{case.description}")

    if error_output:
        sections.append(
            f"## Failing Test Output\n"
            f"When the buggy code is executed against the test suite, "
            f"the following errors occur:\n\n```\n{error_output[:2000]}\n```"
        )

    sections.append(
        "## Instructions\n"
        "Think step by step through the following:\n"
        "1. Read the buggy code carefully and identify the defect\n"
        "2. Explain what the bug is and why it causes incorrect behavior\n"
        "3. Reason about what the correct fix should be\n"
        "4. Provide the corrected function\n\n"
        f"Return the corrected code inside a ```{case.language}``` code block. "
        "Include any necessary import statements at the top. Do not include test cases, "
        "comments, or other code — just the function and its imports."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Repair an original human-induced bug (baseline scenario)
# ---------------------------------------------------------------------------

async def repair_baseline_bug(
    api: APIManager,
    case: BenchmarkCase,
    repairer_key: str,
    eval_dir: str | None = None,
) -> BugEvaluation:
    """Repair a human-induced bug from a benchmark, sample n=10 patches."""
    if eval_dir is None:
        eval_dir = config.BASELINE_DIR

    error_output = ""
    if case.buggy_code and case.test_code:
        buggy_results = evaluate_patches(
            patches=[case.buggy_code],
            program_name=case.name,
            language=case.language,
            test_code=case.test_code,
            checkout_dir=case.checkout_dir,
            source_file_rel=case.source_file_rel,
            required_imports=getattr(case, "required_imports", "") or "",
        )
        if buggy_results:
            error_output = buggy_results[0].test_output or buggy_results[0].error

    prompt = build_baseline_repair_prompt(case, error_output)

    repairer_name = config.MODELS[repairer_key].display_name
    logger.info(
        "Baseline repair: %s with %s — sampling %d patches",
        case.name, repairer_name, config.NUM_SAMPLES,
    )

    responses = await api.generate(
        model_key=repairer_key,
        prompt=prompt,
        temperature=config.REPAIR_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        n_samples=config.NUM_SAMPLES,
        system="You are an expert software engineer specializing in debugging and program repair.",
    )

    patches = [_extract_code_block(resp, case.language) for resp in responses]

    patch_results = evaluate_patches(
        patches=patches,
        program_name=case.name,
        language=case.language,
        test_code=case.test_code,
        checkout_dir=case.checkout_dir,
        source_file_rel=case.source_file_rel,
        required_imports=getattr(case, "required_imports", "") or "",
    )

    evaluation = build_evaluation(
        program_name=case.name,
        language=case.language,
        generator_model="human",
        repairer_model=repairer_key,
        category="human",
        patch_results=patch_results,
        scenario="human",
    )

    save_evaluation(evaluation, out_dir=eval_dir)

    n_passed = sum(1 for r in patch_results if r.passed)
    logger.info(
        "Baseline result: %d/%d patches passed for %s | pass@1=%.2f",
        n_passed, len(patch_results), case.name,
        evaluation.pass_at_k.get(1, 0.0),
    )

    return evaluation


# ---------------------------------------------------------------------------
# Repair a single bug with a single model
# ---------------------------------------------------------------------------

async def repair_bug(
    api: APIManager,
    bug: GeneratedBug,
    repairer_key: str,
    scenario: str = "llm",
    eval_dir: str | None = None,
) -> BugEvaluation:
    """Generate n=10 candidate patches and evaluate them.

    Returns a BugEvaluation with pass@k metrics.
    """
    error_output = bug.validation_output or ""

    prompt = build_repair_prompt(bug, error_output)

    repairer_name = config.MODELS[repairer_key].display_name
    logger.info(
        "Repairing %s (gen: %s, cat: %s) with %s — sampling %d patches",
        bug.program_name, bug.generator_model, bug.category,
        repairer_name, config.NUM_SAMPLES,
    )

    responses = await api.generate(
        model_key=repairer_key,
        prompt=prompt,
        temperature=config.REPAIR_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        n_samples=config.NUM_SAMPLES,
        system="You are an expert software engineer specializing in debugging and program repair.",
    )

    patches: list[str] = []
    for resp in responses:
        code = _extract_code_block(resp, bug.language)
        patches.append(code)

    required_imports = _extract_python_imports(bug.correct_code) if bug.language == "python" else ""
    patch_results = evaluate_patches(
        patches=patches,
        program_name=bug.program_name,
        language=bug.language,
        test_code=bug.test_code,
        checkout_dir=bug.checkout_dir,
        source_file_rel=bug.source_file_rel,
        required_imports=required_imports,
    )

    evaluation = build_evaluation(
        program_name=bug.program_name,
        language=bug.language,
        generator_model=bug.generator_model,
        repairer_model=repairer_key,
        category=bug.category,
        patch_results=patch_results,
        scenario=scenario,
    )

    save_evaluation(evaluation, out_dir=eval_dir)
    _save_patches(bug, repairer_key, patches, patch_results)

    n_passed = sum(1 for r in patch_results if r.passed)
    logger.info(
        "Result: %d/%d patches passed for %s | pass@1=%.2f",
        n_passed, len(patch_results), bug.program_name,
        evaluation.pass_at_k.get(1, 0.0),
    )

    return evaluation


def _save_patches(
    bug: GeneratedBug,
    repairer_key: str,
    patches: list[str],
    results: list[PatchResult],
) -> None:
    """Persist all candidate patches to disk."""
    out_dir = os.path.join(
        config.PATCHES_DIR, bug.generator_model, repairer_key, bug.category
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bug.program_name}.json")
    data = {
        "program_name": bug.program_name,
        "generator": bug.generator_model,
        "repairer": repairer_key,
        "category": bug.category,
        "patches": [
            {
                "code": p,
                "passed": r.passed,
                "error": r.error,
            }
            for p, r in zip(patches, results)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# 7x7 matrix runner
# ---------------------------------------------------------------------------

async def run_repair_matrix(
    api: APIManager,
    bugs: list[GeneratedBug],
    generator_models: list[str] | None = None,
    repairer_models: list[str] | None = None,
    scenario: str = "llm",
    eval_dir: str | None = None,
) -> list[BugEvaluation]:
    """Run the full cross-model repair matrix.

    For each bug (generated by model A), repair with every model B.
    Skips pairs that already have saved evaluations (resumability).
    """
    if eval_dir is None:
        eval_dir = config.EVAL_DIR

    available = api.get_available_models()
    if generator_models is None:
        generator_models = available
    if repairer_models is None:
        repairer_models = available

    evaluations: list[BugEvaluation] = []
    total_pairs = 0
    skipped = 0

    for bug in bugs:
        if bug.generator_model not in generator_models:
            continue
        for repairer in repairer_models:
            if repairer not in available:
                continue
            total_pairs += 1

            eval_path = os.path.join(
                eval_dir, bug.generator_model, repairer,
                f"{bug.program_name}.json",
            )
            if os.path.exists(eval_path):
                logger.debug("Skipping existing evaluation: %s", eval_path)
                skipped += 1
                continue

            try:
                ev = await repair_bug(
                    api, bug, repairer, scenario=scenario, eval_dir=eval_dir,
                )
                evaluations.append(ev)
            except Exception as exc:
                logger.error(
                    "Repair failed for %s (gen=%s, rep=%s): %s",
                    bug.program_name, bug.generator_model, repairer, exc,
                )

    logger.info(
        "Repair matrix complete: %d new evaluations, %d skipped, %d total pairs",
        len(evaluations), skipped, total_pairs,
    )
    api.flush_costs()
    return evaluations
