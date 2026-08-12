"""Tests for QuixBugs test file generation.

Verifies that the dynamic pytest generator produces valid test files that
pass on correct code and fail on buggy code.
"""

import os
import subprocess
import shutil
import sys
import tempfile

import pytest

import config
from dataset_loader import load_quixbugs, _build_quixbugs_test


# Pick 3 well-behaved programs with known JSON test cases
PROGRAMS = ["gcd", "bitcount", "flatten"]


@pytest.fixture(scope="module")
def quixbugs_cases():
    """Load QuixBugs and return a dict keyed by program name."""
    cases = load_quixbugs()
    return {c.name: c for c in cases}


class TestQuixBugsTestGeneration:
    """Verify that generated test files are syntactically valid and functional."""

    @pytest.mark.parametrize("program", PROGRAMS)
    def test_generates_nonempty_test(self, quixbugs_cases, program):
        """Test generator should produce non-empty pytest content."""
        if program not in quixbugs_cases:
            pytest.skip(f"{program} not loaded")
        case = quixbugs_cases[program]
        assert case.test_code.strip(), f"Empty test code for {program}"
        assert "def test_" in case.test_code, f"No test functions for {program}"

    @pytest.mark.parametrize("program", PROGRAMS)
    def test_correct_code_passes(self, quixbugs_cases, program):
        """Correct code should pass all generated tests."""
        if program not in quixbugs_cases:
            pytest.skip(f"{program} not loaded")
        case = quixbugs_cases[program]
        result = _run_in_sandbox(case.correct_code, case.test_code, program)
        assert result.returncode == 0, (
            f"Correct code for {program} FAILED tests:\n{result.stdout}\n{result.stderr}"
        )

    @pytest.mark.parametrize("program", PROGRAMS)
    def test_buggy_code_fails(self, quixbugs_cases, program):
        """Buggy code should fail at least one generated test (or timeout)."""
        if program not in quixbugs_cases:
            pytest.skip(f"{program} not loaded")
        case = quixbugs_cases[program]
        if not case.buggy_code.strip():
            pytest.skip(f"No buggy code for {program}")
        try:
            result = _run_in_sandbox(case.buggy_code, case.test_code, program)
            assert result.returncode != 0, (
                f"Buggy code for {program} PASSED tests (expected failure)"
            )
        except subprocess.TimeoutExpired:
            # Timeout counts as failure — buggy code caused infinite loop
            pass

    def test_build_test_returns_empty_for_missing_program(self):
        """Non-existent program should return empty test code."""
        correct_dir = os.path.join(config.QUIXBUGS_DIR, "correct_python_programs")
        result = _build_quixbugs_test("nonexistent_program_xyz", correct_dir)
        assert result == ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_in_sandbox(
    code: str, test_code: str, program_name: str
) -> subprocess.CompletedProcess:
    """Run code + tests in an isolated temp directory, mimicking the evaluator."""
    tmpdir = tempfile.mkdtemp(prefix="apr_test_")
    try:
        prog_file = os.path.join(tmpdir, f"{program_name}.py")
        test_file = os.path.join(tmpdir, f"test_{program_name}.py")

        with open(prog_file, "w") as f:
            f.write(code)
        with open(test_file, "w") as f:
            f.write(test_code)

        # Copy node.py if needed (graph programs)
        node_src = os.path.join(
            config.QUIXBUGS_DIR, "correct_python_programs", "node.py"
        )
        if os.path.exists(node_src) and "from node import" in test_code:
            shutil.copy2(node_src, os.path.join(tmpdir, "node.py"))

        return subprocess.run(
            [
                sys.executable, "-m", "pytest",
                f"test_{program_name}.py",
                "--timeout=15", "-v", "--tb=short", "--no-header",
                "-p", "no:anyio",
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
