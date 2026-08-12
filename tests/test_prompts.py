"""Tests for prompt template formatters (repair, bug gen, compound injection)."""

import pytest

from bug_generator import (
    GeneratedBug,
    TAXONOMY_PROMPTS,
    COMPOUND_PROMPT_TEMPLATE,
    CATEGORY_EXAMPLES,
    _extract_code_block,
)
from repair_engine import build_repair_prompt, build_baseline_repair_prompt
from dataset_loader import BenchmarkCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CODE = """\
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
"""

SAMPLE_BUGGY_CODE = """\
def gcd(a, b):
    if b == 0:
        return a
    return gcd(a, a % b)
"""

SAMPLE_TEST_CODE = """\
from gcd import gcd

def test_basic():
    assert gcd(10, 5) == 5
"""


@pytest.fixture
def sample_bug():
    return GeneratedBug(
        program_name="gcd",
        language="python",
        generator_model="claude_opus",
        category="variable_misuse",
        correct_code=SAMPLE_CODE,
        buggy_code=SAMPLE_BUGGY_CODE,
        test_code=SAMPLE_TEST_CODE,
        test_path="",
        description="Euclidean GCD algorithm",
    )


@pytest.fixture
def sample_case():
    return BenchmarkCase(
        name="gcd",
        language="python",
        correct_code=SAMPLE_CODE,
        test_code=SAMPLE_TEST_CODE,
        test_path="",
        buggy_code=SAMPLE_BUGGY_CODE,
        description="Euclidean GCD algorithm",
    )


# ---------------------------------------------------------------------------
# Repair prompt tests
# ---------------------------------------------------------------------------

class TestRepairPrompt:
    def test_nonempty(self, sample_bug):
        prompt = build_repair_prompt(sample_bug)
        assert len(prompt) > 100

    def test_contains_buggy_code(self, sample_bug):
        prompt = build_repair_prompt(sample_bug)
        # The buggy code should appear (possibly cleaned)
        assert "gcd" in prompt
        assert "def gcd" in prompt or "gcd(a" in prompt

    def test_contains_category_info(self, sample_bug):
        prompt = build_repair_prompt(sample_bug)
        assert "Variable/Data Misuse" in prompt

    def test_contains_instructions(self, sample_bug):
        prompt = build_repair_prompt(sample_bug)
        assert "Think step by step" in prompt
        assert "```python" in prompt

    def test_includes_description(self, sample_bug):
        prompt = build_repair_prompt(sample_bug)
        assert "Euclidean GCD" in prompt

    def test_includes_error_output(self, sample_bug):
        prompt = build_repair_prompt(sample_bug, error_output="AssertionError: gcd(10,5) != 5")
        assert "AssertionError" in prompt
        assert "Failing Test Output" in prompt

    def test_no_error_output_section_when_empty(self, sample_bug):
        prompt = build_repair_prompt(sample_bug, error_output="")
        assert "Failing Test Output" not in prompt

    def test_truncates_long_error_output(self, sample_bug):
        long_error = "x" * 5000
        prompt = build_repair_prompt(sample_bug, error_output=long_error)
        # Error output should be truncated to 2000 chars
        assert len(prompt) < len(long_error)

    def test_handles_empty_description(self):
        bug = GeneratedBug(
            program_name="test",
            language="python",
            generator_model="gpt53_codex",
            category="logic_error",
            correct_code=SAMPLE_CODE,
            buggy_code=SAMPLE_BUGGY_CODE,
            test_code="",
            test_path="",
            description="",
        )
        prompt = build_repair_prompt(bug)
        assert "Algorithm Description" not in prompt

    def test_handles_empty_buggy_code(self):
        bug = GeneratedBug(
            program_name="test",
            language="python",
            generator_model="gpt53_codex",
            category="logic_error",
            correct_code=SAMPLE_CODE,
            buggy_code="",
            test_code="",
            test_path="",
        )
        prompt = build_repair_prompt(bug)
        assert "```python" in prompt


# ---------------------------------------------------------------------------
# Baseline repair prompt tests
# ---------------------------------------------------------------------------

class TestBaselineRepairPrompt:
    def test_nonempty(self, sample_case):
        prompt = build_baseline_repair_prompt(sample_case)
        assert len(prompt) > 100

    def test_contains_buggy_code(self, sample_case):
        prompt = build_baseline_repair_prompt(sample_case)
        assert "gcd" in prompt

    def test_no_category_info(self, sample_case):
        """Baseline prompt should NOT include taxonomy category."""
        prompt = build_baseline_repair_prompt(sample_case)
        assert "Variable/Data Misuse" not in prompt
        assert "Logic/Condition Error" not in prompt

    def test_includes_error_output(self, sample_case):
        prompt = build_baseline_repair_prompt(sample_case, error_output="FAILED test_basic")
        assert "FAILED test_basic" in prompt


# ---------------------------------------------------------------------------
# Bug generation prompt tests
# ---------------------------------------------------------------------------

class TestBugGenerationPrompts:
    @pytest.mark.parametrize("category", list(TAXONOMY_PROMPTS.keys()))
    def test_all_categories_have_prompts(self, category):
        template = TAXONOMY_PROMPTS[category]
        assert len(template) > 50

    @pytest.mark.parametrize("category", list(TAXONOMY_PROMPTS.keys()))
    def test_prompt_has_placeholders(self, category):
        template = TAXONOMY_PROMPTS[category]
        assert "{language}" in template
        assert "{code}" in template

    @pytest.mark.parametrize("category", list(TAXONOMY_PROMPTS.keys()))
    def test_prompt_formats_without_error(self, category):
        template = TAXONOMY_PROMPTS[category]
        result = template.format(language="python", code=SAMPLE_CODE)
        assert "python" in result
        assert "def gcd" in result

    @pytest.mark.parametrize("category", list(TAXONOMY_PROMPTS.keys()))
    def test_prompt_mentions_one_bug(self, category):
        template = TAXONOMY_PROMPTS[category]
        assert "EXACTLY ONE" in template or "exactly one" in template


# ---------------------------------------------------------------------------
# Compound bug prompt tests
# ---------------------------------------------------------------------------

class TestCompoundBugPrompt:
    @pytest.mark.parametrize("category", list(CATEGORY_EXAMPLES.keys()))
    def test_compound_prompt_formats(self, category):
        result = COMPOUND_PROMPT_TEMPLATE.format(
            language="python",
            category_name="Variable/Data Misuse",
            category_examples=CATEGORY_EXAMPLES[category],
            code=SAMPLE_BUGGY_CODE,
        )
        assert "ALREADY contains" in result
        assert "def gcd" in result
        assert "one additional" in result

    def test_compound_preserves_existing_bug(self):
        result = COMPOUND_PROMPT_TEMPLATE.format(
            language="python",
            category_name="Logic Error",
            category_examples="- test example",
            code=SAMPLE_BUGGY_CODE,
        )
        assert "Keep the existing" in result or "do NOT fix" in result.lower()


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

class TestCodeBlockExtraction:
    def test_extracts_python_block(self):
        response = "Here's the fix:\n```python\ndef gcd(a, b):\n    return a\n```\nDone."
        code = _extract_code_block(response, "python")
        assert "def gcd" in code

    def test_extracts_generic_block(self):
        response = "Fix:\n```\ndef foo():\n    pass\n```"
        code = _extract_code_block(response, "python")
        assert "def foo" in code

    def test_returns_raw_if_no_block(self):
        response = "def gcd(a, b):\n    return a"
        code = _extract_code_block(response, "python")
        assert "def gcd" in code

    def test_empty_response(self):
        assert _extract_code_block("", "python") == ""

    def test_only_backticks(self):
        assert _extract_code_block("```\n```", "python") == ""
