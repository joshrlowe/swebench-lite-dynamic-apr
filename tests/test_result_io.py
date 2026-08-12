"""Tests for result file I/O — save/load round-trips, resumability, corruption handling."""

import json
import os

import pytest

from evaluator import (
    BugEvaluation,
    PatchResult,
    build_evaluation,
    save_evaluation,
)
from bug_generator import GeneratedBug, save_bug, load_bug


# ---------------------------------------------------------------------------
# BugEvaluation save/load round-trip
# ---------------------------------------------------------------------------

class TestEvaluationRoundTrip:
    def test_save_creates_file(self, tmp_path):
        ev = _make_evaluation()
        path = save_evaluation(ev, out_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_save_load_round_trip(self, tmp_path):
        ev = _make_evaluation()
        path = save_evaluation(ev, out_dir=str(tmp_path))

        with open(path) as f:
            data = json.load(f)

        assert data["program_name"] == "gcd"
        assert data["language"] == "python"
        assert data["generator_model"] == "claude_opus"
        assert data["repairer_model"] == "deepseek_r1"
        assert data["category"] == "variable_misuse"
        assert data["scenario"] == "llm"
        assert data["n_samples"] == 3
        assert data["n_correct"] == 2
        # pass_at_k keys are stringified in JSON
        assert "1" in data["pass_at_k"]
        assert isinstance(data["pass_at_k"]["1"], float)

    def test_directory_structure(self, tmp_path):
        """Evaluation file should be at out_dir/generator/repairer/program.json."""
        ev = _make_evaluation()
        path = save_evaluation(ev, out_dir=str(tmp_path))
        expected = os.path.join(
            str(tmp_path), "claude_opus", "deepseek_r1", "gcd.json"
        )
        assert path == expected

    def test_patches_included(self, tmp_path):
        ev = _make_evaluation()
        path = save_evaluation(ev, out_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert len(data["patches"]) == 3
        assert data["patches"][0]["passed"] is True
        assert data["patches"][2]["passed"] is False


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

class TestResumability:
    def test_existing_file_would_be_skipped(self, tmp_path):
        """Simulate the resume check used in repair_engine.run_repair_matrix."""
        ev = _make_evaluation()
        path = save_evaluation(ev, out_dir=str(tmp_path))
        # The pipeline checks os.path.exists(eval_path) before running
        assert os.path.exists(path)

    def test_different_programs_dont_collide(self, tmp_path):
        ev1 = _make_evaluation(program_name="gcd")
        ev2 = _make_evaluation(program_name="bitcount")
        p1 = save_evaluation(ev1, out_dir=str(tmp_path))
        p2 = save_evaluation(ev2, out_dir=str(tmp_path))
        assert p1 != p2
        assert os.path.exists(p1)
        assert os.path.exists(p2)


# ---------------------------------------------------------------------------
# Corruption / edge cases
# ---------------------------------------------------------------------------

class TestCorruptionHandling:
    def test_empty_json_file(self, tmp_path):
        """Loading an empty file should not crash the system."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("")
        with pytest.raises((json.JSONDecodeError, Exception)):
            with open(str(bad_file)) as f:
                json.load(f)

    def test_truncated_json(self, tmp_path):
        """Truncated JSON should raise on load."""
        bad_file = tmp_path / "truncated.json"
        bad_file.write_text('{"program_name": "gcd", "n_samples": ')
        with pytest.raises(json.JSONDecodeError):
            with open(str(bad_file)) as f:
                json.load(f)

    def test_missing_keys_in_json(self, tmp_path):
        """JSON missing expected keys should fail gracefully on field access."""
        bad_file = tmp_path / "incomplete.json"
        bad_file.write_text('{"program_name": "gcd"}')
        with open(str(bad_file)) as f:
            data = json.load(f)
        # Accessing missing key should raise KeyError
        with pytest.raises(KeyError):
            _ = data["n_samples"]


# ---------------------------------------------------------------------------
# GeneratedBug save/load round-trip
# ---------------------------------------------------------------------------

class TestBugRoundTrip:
    def test_save_and_load(self, tmp_path):
        import config
        original_bugs_dir = config.BUGS_DIR
        config.BUGS_DIR = str(tmp_path)
        try:
            bug = _make_bug()
            path = save_bug(bug)
            assert os.path.exists(path)

            loaded = load_bug(path)
            assert loaded.program_name == bug.program_name
            assert loaded.language == bug.language
            assert loaded.generator_model == bug.generator_model
            assert loaded.category == bug.category
            assert loaded.buggy_code == bug.buggy_code
            assert loaded.correct_code == bug.correct_code
            assert loaded.validated == bug.validated
        finally:
            config.BUGS_DIR = original_bugs_dir

    def test_bug_directory_structure(self, tmp_path):
        """Bug file should be at BUGS_DIR/model/category/program.json."""
        import config
        original_bugs_dir = config.BUGS_DIR
        config.BUGS_DIR = str(tmp_path)
        try:
            bug = _make_bug()
            path = save_bug(bug)
            expected = os.path.join(
                str(tmp_path), "claude_opus", "variable_misuse", "gcd.json"
            )
            assert path == expected
        finally:
            config.BUGS_DIR = original_bugs_dir


# ---------------------------------------------------------------------------
# build_evaluation
# ---------------------------------------------------------------------------

class TestBuildEvaluation:
    def test_computes_pass_at_k(self):
        results = [
            PatchResult(patch_code="a", passed=True),
            PatchResult(patch_code="b", passed=True),
            PatchResult(patch_code="c", passed=False),
        ]
        ev = build_evaluation(
            program_name="gcd",
            language="python",
            generator_model="human",
            repairer_model="claude_opus",
            category="human",
            patch_results=results,
        )
        assert ev.n_samples == 3
        assert ev.n_correct == 2
        assert 1 in ev.pass_at_k
        assert ev.pass_at_k[1] == pytest.approx(2 / 3)

    def test_all_pass(self):
        results = [PatchResult(patch_code="x", passed=True) for _ in range(10)]
        ev = build_evaluation("t", "python", "h", "r", "c", results)
        assert ev.pass_at_k[1] == 1.0
        assert ev.pass_at_k[10] == 1.0

    def test_none_pass(self):
        results = [PatchResult(patch_code="x", passed=False) for _ in range(10)]
        ev = build_evaluation("t", "python", "h", "r", "c", results)
        assert ev.pass_at_k[1] == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(program_name: str = "gcd") -> BugEvaluation:
    results = [
        PatchResult(patch_code="def gcd(a,b): return a", passed=True),
        PatchResult(patch_code="def gcd(a,b): return b", passed=True),
        PatchResult(patch_code="def gcd(a,b): return 0", passed=False, error="AssertionError"),
    ]
    return build_evaluation(
        program_name=program_name,
        language="python",
        generator_model="claude_opus",
        repairer_model="deepseek_r1",
        category="variable_misuse",
        patch_results=results,
    )


def _make_bug() -> GeneratedBug:
    return GeneratedBug(
        program_name="gcd",
        language="python",
        generator_model="claude_opus",
        category="variable_misuse",
        correct_code="def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n",
        buggy_code="def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a, a % b)\n",
        test_code="from gcd import gcd\ndef test(): assert gcd(10,5)==5\n",
        test_path="",
        validated=True,
        validation_output="FAILED test_basic",
    )
