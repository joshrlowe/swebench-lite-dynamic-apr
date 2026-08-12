"""Tests for the subprocess pytest sandbox (evaluator._run_python_patch)."""

import pytest

from evaluator import _run_python_patch, _prepend_imports_if_needed, evaluate_patches


# ---------------------------------------------------------------------------
# Simple test code used across sandbox tests
# ---------------------------------------------------------------------------

SIMPLE_TEST = """
from add_nums import add_nums

def test_basic():
    assert add_nums(2, 3) == 5

def test_zero():
    assert add_nums(0, 0) == 0

def test_negative():
    assert add_nums(-1, 1) == 0
"""


class TestSandboxExecution:
    def test_correct_code_passes(self):
        """Good code should produce passed=True."""
        code = "def add_nums(a, b):\n    return a + b\n"
        result = _run_python_patch(code, SIMPLE_TEST, "add_nums")
        assert result.passed is True
        assert result.error == ""

    def test_buggy_code_fails(self):
        """Code with a bug should produce passed=False with output."""
        code = "def add_nums(a, b):\n    return a - b\n"
        result = _run_python_patch(code, SIMPLE_TEST, "add_nums")
        assert result.passed is False
        assert result.test_output or result.error

    def test_syntax_error_fails_gracefully(self):
        """Syntax errors should fail, not crash the sandbox."""
        code = "def add_nums(a, b)\n    return a + b\n"  # missing colon
        result = _run_python_patch(code, SIMPLE_TEST, "add_nums")
        assert result.passed is False

    def test_infinite_loop_times_out(self):
        """Infinite loop should hit timeout and return failure."""
        code = "def add_nums(a, b):\n    while True:\n        pass\n"
        result = _run_python_patch(code, SIMPLE_TEST, "add_nums")
        assert result.passed is False
        assert "Timeout" in result.error or "timeout" in result.test_output.lower()

    def test_empty_patch_fails(self):
        """Empty patch string should be handled via evaluate_patches."""
        results = evaluate_patches(
            patches=[""],
            program_name="add_nums",
            language="python",
            test_code=SIMPLE_TEST,
        )
        assert len(results) == 1
        assert results[0].passed is False
        assert "Empty patch" in results[0].error

    def test_runtime_error_fails(self):
        """Code that raises at runtime should fail, not crash."""
        code = "def add_nums(a, b):\n    raise ValueError('boom')\n"
        result = _run_python_patch(code, SIMPLE_TEST, "add_nums")
        assert result.passed is False


# ---------------------------------------------------------------------------
# Import prepending
# ---------------------------------------------------------------------------

class TestPrependImports:
    def test_prepends_when_missing(self):
        code = "def foo():\n    pass\n"
        result = _prepend_imports_if_needed(code, "import math")
        assert result.startswith("import math")
        assert "def foo" in result

    def test_skips_when_already_has_imports(self):
        code = "import os\ndef foo():\n    pass\n"
        result = _prepend_imports_if_needed(code, "import math")
        # Should NOT prepend since code already starts with import
        assert result == code

    def test_no_op_with_empty_imports(self):
        code = "def foo():\n    pass\n"
        result = _prepend_imports_if_needed(code, "")
        assert result == code

    def test_no_op_with_whitespace_imports(self):
        code = "def foo():\n    pass\n"
        result = _prepend_imports_if_needed(code, "   \n  ")
        assert result == code
