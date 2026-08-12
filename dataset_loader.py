"""Module 2: Dataset ingestion for QuixBugs (Python) and Defects4J (Java)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


def _extract_python_imports(code: str) -> str:
    """Extract top-level import lines from Python code (before first def/class)."""
    imports: list[str] = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith(("import ", "from ")):
            imports.append(line)
        elif s.startswith(("def ", "class ")):
            break
    return "\n".join(imports) if imports else ""


def _clean_buggy_code_for_prompt(code: str, program_name: str) -> str:
    """Strip trailing docstrings and noise from buggy code before sending to model."""
    if not code:
        return code
    lines = code.splitlines()
    result: list[str] = []
    seen_def = False
    for line in lines:
        s = line.strip()
        if s.startswith(("import ", "from ")):
            result.append(line)
        elif s.startswith("def "):
            seen_def = True
            result.append(line)
        elif seen_def and s in ('"""', "'''") and not line.startswith((" ", "\t")):
            break
        elif seen_def or result:
            result.append(line)
    return "\n".join(result).rstrip()


@dataclass
class BenchmarkCase:
    name: str
    language: str
    correct_code: str
    test_code: str
    test_path: str
    buggy_code: str = ""
    description: str = ""
    checkout_dir: str = ""
    source_file_rel: str = ""
    required_imports: str = ""  # Python: imports to prepend if patch omits them


# ---------------------------------------------------------------------------
# QuixBugs (Python, 40 programs)
# ---------------------------------------------------------------------------

QUIXBUGS_REPO = "https://github.com/jkoppel/QuixBugs.git"


def _clone_quixbugs() -> None:
    """Clone QuixBugs repo if not already present."""
    if os.path.isdir(config.QUIXBUGS_DIR):
        logger.info("QuixBugs already cloned at %s", config.QUIXBUGS_DIR)
        return
    os.makedirs(config.BENCHMARKS_DIR, exist_ok=True)
    logger.info("Cloning QuixBugs to %s …", config.QUIXBUGS_DIR)
    subprocess.run(
        ["git", "clone", "--depth", "1", QUIXBUGS_REPO, config.QUIXBUGS_DIR],
        check=True, capture_output=True,
    )


def _build_quixbugs_test(program_name: str, correct_dir: str) -> str:
    """Build a standalone pytest test file for a QuixBugs program.

    QuixBugs stores test data as JSON in json_testcases/ (one JSON array per line).
    Some programs (graph-based) have dedicated _test.py files instead.
    """
    import json

    tc_path = os.path.join(config.QUIXBUGS_DIR, "json_testcases", f"{program_name}.json")

    if not os.path.exists(tc_path):
        test_py = os.path.join(correct_dir, f"{program_name}_test.py")
        if os.path.exists(test_py):
            return _adapt_inline_test(test_py, program_name)
        logger.warning("No test data for %s", program_name)
        return ""

    with open(tc_path) as f:
        raw = f.read().strip()

    cases: list = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not cases:
        logger.warning("Empty test-case file for %s", program_name)
        return ""

    node_import = ""
    node_path = os.path.join(config.QUIXBUGS_DIR, "correct_python_programs", "node.py")
    if os.path.exists(node_path):
        with open(node_path) as f:
            node_src = f.read()
        if _program_uses_node(program_name, correct_dir):
            node_import = node_src + "\n\n"

    normalize_helper = (
        "def _normalize(obj):\n"
        "    if isinstance(obj, (list, tuple)):\n"
        "        return [_normalize(x) for x in obj]\n"
        "    return obj\n\n"
    )

    lines = [
        node_import,
        normalize_helper,
        f"from {program_name} import {program_name}",
        "",
    ]
    for idx, case in enumerate(cases):
        if isinstance(case, list) and len(case) == 2:
            args, expected = case
            if not isinstance(args, list):
                args = [args]
            args_str = ", ".join(repr(a) for a in args)
            lines.append(f"def test_case_{idx}():")
            lines.append(f"    result = {program_name}({args_str})")
            if _is_generator_program(program_name):
                lines.append(f"    assert _normalize(list(result)) == _normalize({expected!r})")
            elif _is_float_program(program_name):
                lines.append(f"    assert abs(result - {expected!r}) < 1e-4")
            else:
                lines.append(f"    assert _normalize(result) == _normalize({expected!r})")
            lines.append("")
        else:
            logger.debug("Skipping malformed test case %d for %s", idx, program_name)

    return "\n".join(lines)


def _program_uses_node(program_name: str, correct_dir: str) -> bool:
    """Check if a program imports the Node class."""
    prog_path = os.path.join(correct_dir, f"{program_name}.py")
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            return "node" in f.read().lower() and "import" in open(prog_path).read().lower()
    return False


GENERATOR_PROGRAMS = {"flatten", "kheapsort", "powerset", "subsequences", "wrap", "sieve"}
FLOAT_PROGRAMS = {"sqrt"}


def _is_generator_program(name: str) -> bool:
    return name in GENERATOR_PROGRAMS


def _is_float_program(name: str) -> bool:
    return name in FLOAT_PROGRAMS


def _adapt_inline_test(test_path: str, program_name: str) -> str:
    """Convert a QuixBugs driver _test.py into a proper standalone pytest file.

    Captures the correct program's output at load time and embeds it as a
    sorted-token reference for comparison, handling dict ordering differences.
    """
    with open(test_path) as f:
        content = f.read()

    content = content.replace("from .node import Node", "from node import Node")
    content = content.replace(
        f"from .{program_name} import", f"from {program_name} import"
    )
    content = content.replace(
        f"from correct_python_programs.{program_name} import",
        f"from {program_name} import",
    )

    # Pre-compute reference output from the correct program
    expected_tokens = _get_correct_output_tokens(test_path, program_name)

    test_code = (
        "import sys\nimport io\n\n"
        + content
        + "\n\ndef test_main():\n"
        "    buf = io.StringIO()\n"
        "    old_stdout = sys.stdout\n"
        "    sys.stdout = buf\n"
        "    try:\n"
        "        main()\n"
        "    finally:\n"
        "        sys.stdout = old_stdout\n"
        "    output = buf.getvalue().strip()\n"
        "    assert len(output) > 0, 'No output produced'\n"
        "    assert 'Error' not in output and 'Traceback' not in output\n"
    )

    if expected_tokens:
        test_code += (
            f"    expected_tokens = {expected_tokens!r}\n"
            "    got_tokens = sorted(output.split())\n"
            "    assert got_tokens == expected_tokens, (\n"
            "        f'Output mismatch (sorted tokens):\\n  got: {got_tokens}\\n  exp: {expected_tokens}'\n"
            "    )\n"
        )

    return test_code


def _get_correct_output_tokens(test_path: str, program_name: str) -> list[str]:
    """Run the inline test against the correct program, return sorted output tokens.

    Returns a sorted list of whitespace-split tokens, or [] on failure.
    """
    import shutil
    import tempfile

    correct_dir = os.path.join(config.QUIXBUGS_DIR, "correct_python_programs")
    correct_src = os.path.join(correct_dir, f"{program_name}.py")
    if not os.path.exists(correct_src):
        return []

    with open(test_path) as f:
        content = f.read()

    content = content.replace("from .node import Node", "from node import Node")
    content = content.replace(
        f"from .{program_name} import", f"from {program_name} import"
    )
    content = content.replace(
        f"from correct_python_programs.{program_name} import",
        f"from {program_name} import",
    )

    tmpdir = tempfile.mkdtemp(prefix="apr_ref_")
    try:
        shutil.copy2(correct_src, os.path.join(tmpdir, f"{program_name}.py"))

        node_src = os.path.join(correct_dir, "node.py")
        if os.path.exists(node_src):
            shutil.copy2(node_src, os.path.join(tmpdir, "node.py"))

        # Strip 'if __name__ == "__main__": main()' to avoid double execution
        cleaned = "\n".join(
            line for line in content.splitlines()
            if not line.strip().startswith("if __name__")
            and not (line.strip() == "main()" and "if __name__" in content)
        )
        runner = (
            "import sys, io\n"
            + cleaned
            + "\nbuf = io.StringIO()\n"
            "old_stdout = sys.stdout\n"
            "sys.stdout = buf\n"
            "try:\n"
            "    main()\n"
            "finally:\n"
            "    sys.stdout = old_stdout\n"
            "print(buf.getvalue().strip())\n"
        )
        runner_path = os.path.join(tmpdir, "_runner.py")
        with open(runner_path, "w") as f:
            f.write(runner)

        result = subprocess.run(
            [sys.executable, runner_path],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return sorted(result.stdout.strip().split())
        logger.debug(
            "Reference run failed for %s: rc=%d %s",
            program_name, result.returncode, result.stderr[:200],
        )
        return []
    except Exception as exc:
        logger.debug("Reference output capture failed for %s: %s", program_name, exc)
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_quixbugs() -> list[BenchmarkCase]:
    """Load all 40 QuixBugs Python programs."""
    _clone_quixbugs()
    correct_dir = os.path.join(config.QUIXBUGS_DIR, "correct_python_programs")
    if not os.path.isdir(correct_dir):
        raise FileNotFoundError(f"QuixBugs correct programs dir not found: {correct_dir}")

    buggy_dir = os.path.join(config.QUIXBUGS_DIR, "python_programs")

    cases: list[BenchmarkCase] = []
    for fname in sorted(os.listdir(correct_dir)):
        if not fname.endswith(".py"):
            continue
        if fname.startswith("__") or fname.endswith("_test.py") or fname == "node.py":
            continue
        name = fname[:-3]
        correct_path = os.path.join(correct_dir, fname)
        with open(correct_path) as f:
            correct_code = f.read()

        buggy_code = ""
        buggy_path = os.path.join(buggy_dir, fname)
        if os.path.exists(buggy_path):
            with open(buggy_path) as f:
                buggy_code = f.read()

        test_code = _build_quixbugs_test(name, correct_dir)
        if not test_code:
            logger.warning("Skipping %s – no tests generated", name)
            continue

        tc_path = os.path.join(config.QUIXBUGS_DIR, "json_testcases", f"{name}.json")
        required_imports = _extract_python_imports(correct_code) if correct_code else ""
        cases.append(BenchmarkCase(
            name=name,
            language="python",
            correct_code=correct_code,
            test_code=test_code,
            test_path=tc_path,
            buggy_code=buggy_code,
            description=f"QuixBugs algorithmic program: {name}",
            required_imports=required_imports,
        ))

    logger.info("Loaded %d QuixBugs Python programs", len(cases))
    return cases


# ---------------------------------------------------------------------------
# Defects4J (Java, 60 bugs)
# ---------------------------------------------------------------------------

DEFECTS4J_REPO = "https://github.com/rjust/defects4j.git"


def _install_defects4j() -> None:
    """Clone and initialize Defects4J if not already present."""
    if os.path.isdir(config.DEFECTS4J_DIR):
        d4j_bin = os.path.join(config.DEFECTS4J_DIR, "framework", "bin", "defects4j")
        if os.path.exists(d4j_bin):
            logger.info("Defects4J already installed at %s", config.DEFECTS4J_DIR)
            return

    os.makedirs(config.BENCHMARKS_DIR, exist_ok=True)
    logger.info("Cloning Defects4J to %s …", config.DEFECTS4J_DIR)
    subprocess.run(
        ["git", "clone", DEFECTS4J_REPO, config.DEFECTS4J_DIR],
        check=True, capture_output=True,
    )
    logger.info("Running Defects4J init.sh (this may take several minutes) …")
    subprocess.run(
        ["bash", "init.sh"],
        cwd=config.DEFECTS4J_DIR,
        check=True, capture_output=True,
        timeout=600,
    )


def _checkout_d4j_bug(
    project: str, bug_id: int, version: str
) -> str:
    """Checkout a Defects4J bug version. Returns the checkout directory path."""
    work_dir = os.path.join(
        config.DEFECTS4J_WORKDIR, f"{project}_{bug_id}{version}"
    )
    if os.path.isdir(work_dir):
        return work_dir

    os.makedirs(config.DEFECTS4J_WORKDIR, exist_ok=True)
    d4j_bin = os.path.join(config.DEFECTS4J_DIR, "framework", "bin", "defects4j")
    subprocess.run(
        [d4j_bin, "checkout", "-p", project, "-v", f"{bug_id}{version}", "-w", work_dir],
        check=True, capture_output=True, timeout=120,
    )
    return work_dir


def _get_modified_sources(project: str, bug_id: int) -> list[str]:
    """Get the list of modified source files for a Defects4J bug."""
    d4j_bin = os.path.join(config.DEFECTS4J_DIR, "framework", "bin", "defects4j")
    work_dir = _checkout_d4j_bug(project, bug_id, "b")

    result = subprocess.run(
        [d4j_bin, "export", "-p", "classes.modified"],
        cwd=work_dir,
        capture_output=True, text=True, timeout=30,
    )
    classes = result.stdout.strip().split("\n")

    src_dir_result = subprocess.run(
        [d4j_bin, "export", "-p", "dir.src.classes"],
        cwd=work_dir,
        capture_output=True, text=True, timeout=30,
    )
    src_dir = src_dir_result.stdout.strip()

    source_files = []
    for cls in classes:
        rel_path = os.path.join(src_dir, cls.replace(".", "/") + ".java")
        source_files.append(rel_path)

    return source_files


def load_defects4j() -> list[BenchmarkCase]:
    """Load the pre-selected 60 Defects4J Java bugs."""
    _install_defects4j()

    cases: list[BenchmarkCase] = []
    for project, bug_ids in config.DEFECTS4J_BUGS.items():
        for bug_id in bug_ids:
            try:
                fixed_dir = _checkout_d4j_bug(project, bug_id, "f")
                buggy_dir = _checkout_d4j_bug(project, bug_id, "b")
                source_files = _get_modified_sources(project, bug_id)

                if not source_files:
                    logger.warning("No modified sources for %s-%d, skipping", project, bug_id)
                    continue

                src_rel = source_files[0]
                fixed_src = os.path.join(fixed_dir, src_rel)
                if not os.path.exists(fixed_src):
                    logger.warning("Fixed source not found: %s", fixed_src)
                    continue

                with open(fixed_src) as f:
                    correct_code = f.read()

                buggy_code = ""
                buggy_src = os.path.join(buggy_dir, src_rel)
                if os.path.exists(buggy_src):
                    with open(buggy_src) as f:
                        buggy_code = f.read()

                cases.append(BenchmarkCase(
                    name=f"{project}_{bug_id}",
                    language="java",
                    correct_code=correct_code,
                    test_code="",
                    test_path="",
                    buggy_code=buggy_code,
                    description=f"Defects4J {project} bug #{bug_id}",
                    checkout_dir=buggy_dir,
                    source_file_rel=src_rel,
                ))
            except Exception as exc:
                logger.error("Failed to load %s-%d: %s", project, bug_id, exc)
                continue

    logger.info("Loaded %d Defects4J Java bugs", len(cases))
    return cases


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_all() -> list[BenchmarkCase]:
    """Load all benchmarks (QuixBugs + Defects4J)."""
    cases = load_quixbugs()
    try:
        cases.extend(load_defects4j())
    except Exception as exc:
        logger.warning(
            "Defects4J loading failed (Java may not be installed): %s — "
            "continuing with QuixBugs only.",
            exc,
        )
    return cases
