from __future__ import annotations

import ast, difflib, glob, json, logging, math, os, re, shutil, subprocess, tempfile, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from models import ModelClient

logger = logging.getLogger(__name__)

# Encapsulation for evaluating one patch against the SWEbench harness
@dataclass
class SWEBenchResult:
    resolved: bool = False
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    error: str | None = None
    log_path: str | None = None

# A class wrapper around the SWE-bench Lite harness for running patch evaluations on Docker
class SWEBenchRunner:

    def __init__(self, dataset_name="princeton-nlp/SWE-bench_Lite", timeout=600, cache_level="env", clean=True, log_dir=None):
        self.dataset_name = dataset_name
        self.timeout = timeout
        self.cache_level = cache_level
        self.clean = clean
        self.log_dir = log_dir or str(Path("logs") / "swebench_eval")

    # Apply a candidate patch and run the corresponding SWE-bench test suite
    def evaluate_patch(self, instance_id, patch, run_id=None):
        from swebench.harness.run_evaluation import main as swe_main
        if run_id is None:
            run_id = f"apr_{uuid.uuid4().hex[:8]}"
        if not patch or not patch.strip():
            return SWEBenchResult(resolved=False, error="Empty patch")
        result = SWEBenchResult()
        with tempfile.TemporaryDirectory(prefix="apr_swe_") as tmpdir:
            pred_path = os.path.join(tmpdir, "predictions.jsonl")
            with open(pred_path, "w") as f:
                json.dump({"instance_id": instance_id, "model_name_or_path": "apr_candidate", "model_patch": patch}, f)
                f.write("\n")
            report_path = swe_main(dataset_name=self.dataset_name, split="test", instance_ids=[instance_id], predictions_path=pred_path, max_workers=1, force_rebuild=False, cache_level=self.cache_level, clean=self.clean, open_file_limit=4096, run_id=run_id, timeout=self.timeout, namespace=None, rewrite_reports=False, modal=False, report_dir=self.log_dir)
            result = self._parse_instance_report(instance_id, run_id)
        self._cleanup_instance_images(instance_id)
        return result

    # Run the reference patch to sanity-check the harness
    def evaluate_gold(self, instance_id):
        from swebench.harness.run_evaluation import main as swe_main
        run_id = f"gold_{uuid.uuid4().hex[:8]}"
        swe_main(dataset_name=self.dataset_name, split="test", instance_ids=[instance_id], predictions_path="gold", max_workers=1, force_rebuild=False, cache_level=self.cache_level, clean=self.clean, open_file_limit=4096, run_id=run_id, timeout=self.timeout, namespace=None, rewrite_reports=False, modal=False, report_dir=self.log_dir)
        result = self._parse_instance_report(instance_id, run_id, model_name="gold")
        self._cleanup_instance_images(instance_id)
        return result

    # Find and parse the JSON report that swebench writes after evaluation
    def _parse_instance_report(self, instance_id, run_id, model_name="apr_candidate"):
        candidates = [ Path("logs") / "run_evaluation" / run_id / model_name / instance_id / "report.json", Path(self.log_dir) / "logs" / "run_evaluation" / run_id / model_name / instance_id / "report.json", Path(self.log_dir) / run_id / model_name / instance_id / "report.json", ]
        report_file = None
        for c in candidates:
            if c.exists():
                report_file = c
                break
        if report_file is None:
            return SWEBenchResult(error=f"Report not found; searched: {[str(c) for c in candidates]}")
        report = json.loads(report_file.read_text())
        ftp = report.get(instance_id, {})
        resolved = ftp.get("resolved", False)
        ftp_results = ftp.get("tests_status", {})
        ftp_passed = sum(1 for v in ftp_results.values() if v == "PASSED")
        ptp_status = ftp.get("tests_status_pass_to_pass", {})
        ptp_passed = sum(1 for v in ptp_status.values() if v == "PASSED")
        return SWEBenchResult(resolved=resolved, fail_to_pass_passed=ftp_passed, fail_to_pass_total=len(ftp_results), pass_to_pass_passed=ptp_passed, pass_to_pass_total=len(ptp_status), log_path=str(report_file))

    # Remove Docker images left behind by the eval for this instance
    def _cleanup_instance_images(self, instance_id):
        safe_id = instance_id.replace("/", "_").replace(":", "_")
        try:
            result = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", f"reference=*{safe_id}*"], capture_output=True, text=True, timeout=30)
            for img in result.stdout.strip().splitlines():
                if img and safe_id.lower() in img.lower():
                    subprocess.run(["docker", "rmi", "-f", img], capture_output=True, timeout=30)
        except Exception:
            pass

    # Remove dangling Docker resources to free disk space
    @staticmethod
    def docker_prune():
        try:
            subprocess.run(["docker", "system", "prune", "-f"], capture_output=True, timeout=60)
        except Exception:
            pass

think_regex = re.compile(r"<think>.*?</think>", re.DOTALL)
python_regex = re.compile(r"```[Pp]ython\s*\n(.*?)```", re.DOTALL)
diff_regex = re.compile(r"```(?:diff|patch|unified)\s*\n(.*?)```", re.DOTALL)
plain_regex = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
unclosed_fence_start = re.compile(r"^```(?:[Pp]ython)?\s*\r?\n(.*)$", re.DOTALL)

# Gather all code blocks from the text
def _collect_blocks_in_order(text: str) -> list[str]:
    tagged = [(m.start(), m.group(1)) for m in python_regex.finditer(text)]
    plain = [(m.start(), m.group(1)) for m in plain_regex.finditer(text)]
    combined = sorted(tagged + plain, key=lambda t: t[0])
    return [code for _, code in combined]

# Checks to see if  text looks like a unified diff
def _looks_like_diff(text: str) -> bool:
    lines = text.strip().splitlines()
    has_header = any(l.startswith("---") for l in lines[:5])
    has_hunk = any(l.startswith("@@") for l in lines[:10])
    has_diff_git = any(l.startswith("diff --git") for l in lines[:3])
    return (has_header and has_hunk) or has_diff_git

# Pull Python source code out of an LLM response
def extract_code(text: str) -> str:
    cleaned = think_regex.sub("", text)
    if _collect_blocks_in_order(cleaned):
        return blocks[-1].strip()

    stripped = cleaned.strip()
    um = unclosed_fence_start.match(stripped)
    if um and "```" not in um.group(1):
        candidate = um.group(1).strip()
        if candidate:
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                pass

    if stripped:
        ast.parse(stripped)
        return stripped

# pull a unified diff out of an LLM response
def extract_patch(text: str) -> str:
    cleaned = think_regex.sub("", text)

    diff_blocks = diff_regex.findall(cleaned)
    if diff_blocks:
        return diff_blocks[-1].strip()

    plain_blocks = _collect_blocks_in_order(cleaned)
    for block in reversed(plain_blocks):
        if _looks_like_diff(block):
            return block.strip()

    stripped = cleaned.strip()
    if _looks_like_diff(stripped):
        lines = stripped.splitlines()
        diff_lines = []
        in_diff = False
        for line in lines:
            if line.startswith("---") or line.startswith("diff --git"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
        if diff_lines:
            return "\n".join(diff_lines)

            raise ValueError(f"No unified diff found in response (length={len(text)})")

# try to get a patch first; if that fails, diff the extracted code instead
def extract_patch_or_code(text: str, original_file: str, file_path: str) -> str:
    try:
        return extract_patch(text)
    except ValueError:
        pass

    try:
        modified = extract_code(text)
        return full_file_to_patch(original_file, modified, file_path)
    except ValueError:
        pass

        raise ValueError(f"Could not extract patch or code from response (length={len(text)})")

# diff two full file contents into a unified patch string
def full_file_to_patch(original: str, modified: str, file_path: str) -> str:
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    if not orig_lines or not orig_lines[-1].endswith("\n"):
        orig_lines = [l + "\n" if not l.endswith("\n") else l for l in original.splitlines()]
    if not mod_lines or not mod_lines[-1].endswith("\n"):
        mod_lines = [l + "\n" if not l.endswith("\n") else l for l in modified.splitlines()]

        diff = difflib.unified_diff(orig_lines, mod_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}")
    result = "".join(diff)
    if not result:
        raise ValueError("No differences between original and modified file")
    return result

# Apply a unified diff to a file's content
def apply_patch_to_content(original: str, patch_text: str, file_path: str) -> str:
    with tempfile.TemporaryDirectory(prefix="apr_apply_") as tmpdir:
        orig_path = os.path.join(tmpdir, os.path.basename(file_path))
        patch_path = os.path.join(tmpdir, "change.patch")
        with open(orig_path, "w", encoding="utf-8") as f:
            f.write(original)
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch_text)

        for strip in ("1", "0"):
            result = subprocess.run(["patch", f"-p{strip}", "--batch", "--silent", orig_path, patch_path], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                with open(orig_path, encoding="utf-8") as f:
                    return f.read()
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(original)

                raise ValueError(f"Failed to apply patch to {file_path}: {result.stderr.strip() or result.stdout.strip()}")

# Get every code block from the response as a list of strings
def extract_all_code_blocks(text: str) -> list[str]:
    cleaned = think_regex.sub("", text)
    return [b.strip() for b in _collect_blocks_in_order(cleaned)]

bug_types = [ "variable_naming", "logical_conditional", "loop_iteration", "function_parameter", "off_by_one", "null_reference", ]

BUG_TYPE_DEFINITIONS = { "variable_naming": "A variable is incorrectly spelled or assigned to a different variable than intended.", "logical_conditional": "An incorrect logical operator is used in a conditional statement, e.g, 'and' changed to 'or', '>' changed to '>=', etc.", "loop_iteration": "Incorrect loop boundaries, loop operation on a list, or incorrect termination condition for a loop.", "function_parameter": "Arguments to a function call are passed in the wrong order, the wrong argument is used.", "off_by_one": "An arithmetic expression uses +1 or -1 incorrectly in a boundary calculation or index", "null_reference": "A variable that may be null is dereferenced without a check beforehand", }

repair_template = """\
You are an expert software engineer. We have the following issue in a project:

{problem_statement}

This is where the buggy file is in our software: {buggy_file_path}

And here is its content:

{buggy_file_content}

The following tests fail on the current code and must pass after your fix:
{failing_tests}


Identify and fix the root cause of the bug described in our problem. Your fix should follow best practices and not have any workaronds. Return your fix as a unified diff patch, which can be obtained by using the diff -u command. The patch needs to apply cleanly to the file above such that it parses. Here's the format for the diff:

```diff
[your unified diff patch here]
```"""

injection_template = """\
You are a software testing expert. I need you to introduce a subtle, realistic, and deceptive bug into the following file: {file_path}

Here is the content of the file:

{file_content}

I need you to inject one bug of type {bug_type} into the file. The bug must be subtle and realistic, looking like a human mistake that any software developer could make, even the most experienced. The code must be syntactically valid after this bug is injected. Additionally, don't remove or rename existing functions or classes.

More specifcially, here is the bug type's definition: 
{bug_type_definition}

The following tests must fail after the bug is injectced:

{test_names}

Return the changes as a unified diff patch, which can be produced using the diff -u command:

```diff
[your unified diff patch here]
```"""

compound_introduction = """
The code below is incorrect, as it fails some tests because of an existing defect. I need you to introduce one more bug of the specified type in addition to the existing faulty code.Don't fix or remove the existing bug. The final code must retain syntactically valid and the tests must still fail.
"""

compound_template = compound_introduction + injection_template

# Create the repair prompt with the issue, buggy code, and failing tests
def build_swebench_repair_prompt(problem_statement: str, buggy_file_path: str, buggy_file_content: str, fail_to_pass: list[str] | str) -> str:
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    failing_tests = "\n".join(f"- {t}" for t in fail_to_pass)

    return repair_template.format(problem_statement=problem_statement, buggy_file_path=buggy_file_path, buggy_file_content=buggy_file_content, failing_tests=failing_tests)

# Simplified repair prompt for standalone usage
def build_repair_prompt(buggy_code: str, test_suite: str, failing_tests: str = "") -> str:
    return repair_template.format(problem_statement="(See buggy code and tests below)", buggy_file_path="solution.py", buggy_file_content=buggy_code, failing_tests=failing_tests or "(Run the test suite to identify failing tests)")

# Build the repair prompt, pulling failing test names from a test result object
def build_repair_prompt_with_result(buggy_code: str, test_suite: str, test_result) -> str:
    if hasattr(test_result, "failing_test_names") and test_result.failing_test_names:
        failing_tests = "\n".join(test_result.failing_test_names)
    else:
        failing_tests = ""
    return build_repair_prompt(buggy_code, test_suite, failing_tests)

# Build the prompt that asks an LLM to inject a specific bug type
def build_swebench_injection_prompt(file_content: str, file_path: str, bug_type: str, test_names: list[str] | str, *, mode: str = "single") -> str:
    if bug_type not in BUG_TYPE_DEFINITIONS:
        raise ValueError(f"Unknown bug_type: {bug_type!r}. Must be one of: {bug_types}")
    if mode not in ("single", "compound"):
        raise ValueError(f"mode must be 'single' or 'compound', got {mode!r}")

    test_names_str = "\n".join(f"- {t}" for t in test_names) if isinstance(test_names, list) else "\n".join(f"- {t}" for t in json.loads(test_names))
    tmpl = compound_template if mode == "compound" else injection_template
    return tmpl.format(bug_type=bug_type, bug_type_definition=BUG_TYPE_DEFINITIONS[bug_type], file_path=file_path, file_content=file_content, test_names=test_names_str)

# Shorthand injection prompt for standalone use outside SWE-bench
def build_injection_prompt(source_code: str, test_suite: str, bug_type: str, *, mode: str = "single") -> str:
    return build_swebench_injection_prompt(file_content=source_code, file_path="solution.py", bug_type=bug_type, test_names=["(see test suite)"], mode=mode)

# Compute unbiased pass@k estimate given n samples with c correct
def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n})")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# Build the filesystem path for a raw response JSON file
def raw_path(root: str | Path, *, model_name: str, case_type: str, case_id: int | str, attempt: int) -> Path:
    return (Path(root) / model_name / case_type / str(case_id) / f"{attempt}.json")

# Persist one raw LLM response to disk
def save_raw_response(*, root: str | Path, model_name: str, case_type: str, case_id: int | str, attempt: int, prompt: str, response: Any, display_name: str | None = None, extra: dict | None = None) -> Path:
    path = raw_path(root, model_name=model_name, case_type=case_type, case_id=case_id, attempt=attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = { "model": model_name, "case_id": case_id, "case_type": case_type, "display_name": display_name, "attempt": attempt, "prompt": prompt, "response": response, }
    if extra:
        payload["extra"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

CaseType = Literal["human_written_bugs", "llm_injected_bugs", "compound_bugs"]

# everything needed to attempt one repair: the bug, the context, the tests
@dataclass(frozen=True)
class SWEBenchRepairCase:
    instance_id: str
    case_id: int
    case_type: CaseType
    buggy_file_path: str
    buggy_file_content: str
    problem_statement: str
    fail_to_pass: list[str]
    display_name: str
    source_model_name: str | None = None
    repo_file_content: str | None = None

# formatted UTC timestamp for console output
def _ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")

# Wrap raw case dicts into SWEBenchRepairCase objects for human written bugs
def human_written_bug_cases(cases: list[dict]) -> list[SWEBenchRepairCase]:
    result = []
    for r in cases:
        ftp = json.loads(r["fail_to_pass"]) if isinstance(r["fail_to_pass"], str) else r["fail_to_pass"]
        result.append(SWEBenchRepairCase(instance_id=r["instance_id"], case_id=r["id"], case_type="human_written_bugs", buggy_file_path=r["buggy_file_path"], buggy_file_content=r["buggy_file_content"], problem_statement=r["problem_statement"], fail_to_pass=ftp, display_name=r["instance_id"], source_model_name=None))
    return result

# Read injection JSONs from disk and build repair cases from valid ones
def load_injections(injections_dir: str, seed_source: str, base_cases: list[dict]) -> list[SWEBenchRepairCase]:
    base_lookup: dict[int, dict] = {c["id"]: c for c in base_cases}
    result = []
    for path in glob.glob(os.path.join(injections_dir, "*.json")):
        with open(path) as f:
            inj = json.load(f)
        if not inj.get("valid"):
            continue
        if inj.get("seed_source", "correct") != seed_source:
            continue
        base = base_lookup.get(inj["base_case_id"])
        if base is None:
            continue
        ftp = json.loads(base["fail_to_pass"]) if isinstance(base["fail_to_pass"], str) else base["fail_to_pass"]
        case_type: CaseType = "llm_injected_bugs" if seed_source == "correct" else "compound_bugs"
        tag = "inj" if seed_source == "correct" else "compound"
        inj_id = inj.get("id", Path(path).stem)
        result.append(SWEBenchRepairCase(instance_id=base["instance_id"], case_id=inj_id, case_type=case_type, buggy_file_path=base["buggy_file_path"], buggy_file_content=inj["injected_code"], problem_statement=base["problem_statement"], fail_to_pass=ftp, display_name=f"{base['instance_id']}#{tag}{inj_id}", source_model_name=inj.get("injecting_model"), repo_file_content=base["buggy_file_content"]))
    return result

# Load single-fault injected bug cases for llm injection condition
def llm_injected_bug_cases(injections_dir: str, base_cases: list[dict]) -> list[SWEBenchRepairCase]:
    return load_injections(injections_dir, "correct", base_cases)

# Load compound bug cases for compound condition
def compound_bug_cases(injections_dir: str, base_cases: list[dict]) -> list[SWEBenchRepairCase]:
    return load_injections(injections_dir, "human_buggy", base_cases)

# Extract a patch relative to the original repo content, not the injected version
def extract_repo_relative_patch(response_text: str, injected_content: str, repo_content: str, file_path: str) -> str:
    repair_diff = extract_patch(response_text)
    repaired = apply_patch_to_content(injected_content, repair_diff, file_path)

    return full_file_to_patch(repo_content, repaired, file_path)

# This is our main repair loop where we prompt the model, extract patches, etc.
def run_swebench_repair_loop(case: SWEBenchRepairCase, repairing_model: ModelClient, runner: SWEBenchRunner, *, n_samples: int | None = None, raw_output_dir: str | Path = "results/raw") -> dict:
    if n_samples is None:
        from models import load_config
        n_samples = load_config().experiment.n_samples

        summary = { "attempted": 0, "resolved": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, }

        prompt = build_swebench_repair_prompt(case.problem_statement, case.buggy_file_path, case.buggy_file_content, case.fail_to_pass)

    try:
        print(f"[{_ts()}] [repair {repairing_model.model_name}] {case.display_name}: calling API (n={n_samples})...", flush=True)
        responses = repairing_model.complete_with_retry(prompt, n=n_samples)
        summary["prompt_tokens"] = sum(r.prompt_tokens for r in responses)
        summary["completion_tokens"] = sum(r.completion_tokens for r in responses)
        print(f"[{_ts()}] [repair {repairing_model.model_name}] {case.display_name}: got {len(responses)} response(s), evaluating...", flush=True)

        raw_root = Path(raw_output_dir)

        for attempt_num, response in enumerate(responses):
            save_raw_response(root=raw_root, model_name=repairing_model.model_name, case_type=case.case_type, case_id=case.case_id, attempt=attempt_num, prompt=prompt, response=response.raw_response, display_name=case.display_name)

            patch = None
            resolved = False
            attempt_start = time.time()

            try:
                if case.repo_file_content is not None:
                    patch = extract_repo_relative_patch(response.text, case.buggy_file_content, case.repo_file_content, case.buggy_file_path)
                else:
                    patch = extract_patch_or_code(response.text, case.buggy_file_content, case.buggy_file_path)
                result = runner.evaluate_patch(case.instance_id, patch)
                resolved = result.resolved
            except ValueError as e:
                logger.warning("[%s] %s attempt %d: extraction failed: %s", repairing_model.model_name, case.display_name, attempt_num, e)
            except Exception as e:
                logger.error("[%s] %s attempt %d: eval error: %s", repairing_model.model_name, case.display_name, attempt_num, e)

            summary["attempted"] += 1
            if resolved:
                summary["resolved"] += 1

    except Exception as e:
        logger.error("[%s] %s: %s: %s", repairing_model.model_name, case.display_name, type(e).__name__, e)
        summary["errors"] = 1

    tag = "resolved={}/{}".format(summary['resolved'], summary['attempted'])
    print(f"[{_ts()}] [repair {repairing_model.model_name}] {case.display_name}: done ({tag})", flush=True)
    return summary

def _extract_injection(response_text: str, source_code: str, file_path: str, attempt_num: int) -> tuple[str | None, str | None]:
    try:
        diff = extract_patch(response_text)
        injected = apply_patch_to_content(source_code, diff, file_path)
        if injected.strip() == source_code.strip():
            logger.warning("Attempt %d: diff applied but code unchanged", attempt_num)
            return None, None
        return injected, diff
    except ValueError as e:
        logger.debug("Attempt %d: diff extraction path failed: %s", attempt_num, e)

    try:
        injected = extract_code(response_text)
        if injected.strip() == source_code.strip():
            logger.warning("Attempt %d: code unchanged", attempt_num)
            return None, None
        min_len = int(len(source_code) * 0.5)
        if len(injected) < min_len:
            logger.warning("Attempt %d: extracted code too short (%d vs %d original)", attempt_num, len(injected), len(source_code))
            return None, None
        diff = full_file_to_patch(source_code, injected, file_path)
        return injected, diff
    except ValueError as e:
        logger.warning("Attempt %d: extraction failed: %s", attempt_num, e)

    return None, None

# ask a model to inject a bug, validate it breaks tests, and save results
def inject_bug(model: ModelClient, case: dict, bug_type: str, runner: SWEBenchRunner, *, compound: bool = False, max_retries: int | None = None, raw_output_dir: str = "results/raw", output_dir: str = "results/injections") -> bool:
    display = case.get("instance_id", str(case.get("id", "?")))
    if max_retries is None:
        from models import load_config
        max_retries = load_config().execution.max_injection_retries

    tag_prefix = "compound" if compound else "inject"
    seed_source = "human_buggy" if compound else "correct"
    mode = "compound" if compound else "single"
    case_type = "compound_bugs" if compound else "llm_injected_bugs"
    source_code = case["buggy_file_content"] if compound else case["correct_file_content"]
    file_path = case["buggy_file_path"]
    fail_to_pass = json.loads(case["fail_to_pass"]) if isinstance(case["fail_to_pass"], str) else case["fail_to_pass"]

    prompt = build_swebench_injection_prompt(source_code, file_path, bug_type, fail_to_pass, mode=mode)

    injected_code = None
    injection_patch = None
    valid = False
    attempt = 0

    for attempt in range(max_retries):
        try:
            print(f"[{_ts()}] [{tag_prefix} {model.model_name}] {display} x {bug_type}: attempt {attempt+1}/{max_retries}", flush=True)
            responses = model.complete_with_retry(prompt, n=1)
            response = responses[0]

            display_suffix = f"#{bug_type}_compound" if compound else f"#{bug_type}"
            save_raw_response(root=raw_output_dir, model_name=model.model_name, case_type=case_type, case_id=case["id"], attempt=attempt, prompt=prompt, response=response.raw_response, display_name=f"{display}{display_suffix}", extra={"bug_type": bug_type, "seed_source": seed_source})

            injected_code, injection_diff = _extract_injection(response.text, source_code, file_path, attempt + 1)
            if injected_code is None:
                continue

                validation_patch = full_file_to_patch(case["buggy_file_content"], injected_code, file_path)

            result = runner.evaluate_patch(case["instance_id"], validation_patch)
            if result.error and "Patch Apply Failed" in result.error:
                logger.warning("Attempt %d: validation patch failed to apply", attempt + 1)
                injected_code = None
                continue
            if not result.resolved:
                injection_patch = validation_patch
                valid = True
                print(f"[{_ts()}] [{tag_prefix} {model.model_name}] {display} x {bug_type}: valid", flush=True)
                break
            else:
                logger.warning("Attempt %d: injected code still passes all tests", attempt + 1)

        except Exception as e:
            logger.error("Attempt %d: error: %s: %s", attempt + 1, type(e).__name__, e)

    os.makedirs(output_dir, exist_ok=True)
    result_data = { "base_case_id": case["id"], "instance_id": case["instance_id"], "injecting_model": model.model_name, "bug_type": bug_type, "seed_source": seed_source, "injected_code": injected_code, "injection_patch": injection_patch, "valid": valid, "injection_attempts": min(attempt + 1, max_retries), "buggy_file_path": case["buggy_file_path"], "buggy_file_content": case["buggy_file_content"], "correct_file_content": case.get("correct_file_content"), "problem_statement": case["problem_statement"], "fail_to_pass": case["fail_to_pass"], }
    filename = f"{case['instance_id']}_{model.model_name}_{bug_type}_{seed_source}.json"
    filename = filename.replace("/", "_")
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(result_data, f, indent=2)

    status = "valid" if valid else "INVALID"
    print(f"[{_ts()}] [{tag_prefix} {model.model_name}] {display} x {bug_type}: {status} after {min(attempt+1, max_retries)} attempt(s)", flush=True)
    return valid
