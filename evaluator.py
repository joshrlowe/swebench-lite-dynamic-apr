"""Module 5: Execution sandbox and pass@k estimator."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    patch_code: str
    passed: bool
    test_output: str = ""
    error: str = ""


@dataclass
class BugEvaluation:
    program_name: str
    language: str
    generator_model: str
    repairer_model: str
    category: str
    n_samples: int = 0
    n_correct: int = 0
    pass_at_k: dict[int, float] = field(default_factory=dict)
    patch_results: list[PatchResult] = field(default_factory=list)
    scenario: str = "llm"  # "human", "llm", or "compound"


# ---------------------------------------------------------------------------
# pass@k estimator (Chen et al. 2021, unbiased)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased pass@k estimator.

    n: total samples, c: number that passed, k: metric level.
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if c < 0 or c > n:
        raise ValueError(f"c={c} must be in [0, n={n}]")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_batch(
    results: list[tuple[int, int]],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Average pass@k across multiple programs.

    *results* is a list of (n, c) tuples – one per program.
    Returns {k: average_pass_at_k}.
    """
    if k_values is None:
        k_values = config.PASS_K_VALUES
    out: dict[int, float] = {}
    for k in k_values:
        scores = [pass_at_k(n, c, k) for n, c in results if n >= k]
        out[k] = float(np.mean(scores)) if scores else 0.0
    return out


# ---------------------------------------------------------------------------
# Python sandbox execution (pytest)
# ---------------------------------------------------------------------------

def _prepend_imports_if_needed(patch_code: str, required_imports: str) -> str:
    """Prepend required imports to patch if it doesn't already have them."""
    if not required_imports.strip():
        return patch_code
    first_line = patch_code.strip().split("\n")[0].strip() if patch_code.strip() else ""
    if first_line.startswith(("import ", "from ")):
        return patch_code
    return required_imports.rstrip() + "\n\n" + patch_code.lstrip()


def _run_python_patch(
    patch_code: str,
    test_code: str,
    program_name: str,
    required_imports: str = "",
) -> PatchResult:
    """Execute a Python patch against its test suite in an isolated temp dir."""
    patch_code = _prepend_imports_if_needed(patch_code, required_imports)
    tmpdir = tempfile.mkdtemp(prefix="apr_py_")
    try:
        prog_file = os.path.join(tmpdir, f"{program_name}.py")
        test_file = os.path.join(tmpdir, f"test_{program_name}.py")

        with open(prog_file, "w") as f:
            f.write(patch_code)
        with open(test_file, "w") as f:
            f.write(test_code)

        node_src = os.path.join(config.QUIXBUGS_DIR, "correct_python_programs", "node.py")
        if os.path.exists(node_src) and "from node import" in test_code:
            shutil.copy2(node_src, os.path.join(tmpdir, "node.py"))

        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                f"test_{program_name}.py",
                f"--timeout={config.TEST_TIMEOUT_SECONDS}",
                "-v", "--tb=short", "--no-header",
                "-p", "no:anyio",
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=config.TEST_TIMEOUT_SECONDS + 5,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return PatchResult(
            patch_code=patch_code, passed=passed, test_output=output.strip()
        )
    except subprocess.TimeoutExpired:
        return PatchResult(
            patch_code=patch_code, passed=False, error="Timeout"
        )
    except Exception as exc:
        return PatchResult(
            patch_code=patch_code, passed=False, error=str(exc)
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Java sandbox execution (Defects4J)
# ---------------------------------------------------------------------------

def _run_java_patch(
    patch_code: str,
    checkout_dir: str,
    source_file_rel: str,
) -> PatchResult:
    """Apply a Java patch to a Defects4J checkout and run its tests."""
    src_path = os.path.join(checkout_dir, source_file_rel)
    backup_path = src_path + ".bak"

    try:
        if os.path.exists(src_path):
            shutil.copy2(src_path, backup_path)

        with open(src_path, "w") as f:
            f.write(patch_code)

        d4j_bin = os.path.join(config.DEFECTS4J_DIR, "framework", "bin", "defects4j")
        result = subprocess.run(
            [d4j_bin, "test"],
            cwd=checkout_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + "\n" + result.stderr
        passed = "Failing tests: 0" in output
        return PatchResult(
            patch_code=patch_code, passed=passed, test_output=output.strip()
        )
    except subprocess.TimeoutExpired:
        return PatchResult(patch_code=patch_code, passed=False, error="Timeout")
    except Exception as exc:
        return PatchResult(patch_code=patch_code, passed=False, error=str(exc))
    finally:
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, src_path)
            os.remove(backup_path)


# ---------------------------------------------------------------------------
# Unified patch evaluator
# ---------------------------------------------------------------------------

def evaluate_patches(
    patches: list[str],
    program_name: str,
    language: str,
    test_code: str = "",
    checkout_dir: str = "",
    source_file_rel: str = "",
    required_imports: str = "",
) -> list[PatchResult]:
    """Evaluate a list of candidate patches, return PatchResult per patch."""
    results: list[PatchResult] = []
    for i, patch in enumerate(patches):
        if not patch.strip():
            results.append(PatchResult(patch_code=patch, passed=False, error="Empty patch"))
            continue
        logger.debug("Evaluating patch %d/%d for %s", i + 1, len(patches), program_name)
        if language == "python":
            r = _run_python_patch(patch, test_code, program_name, required_imports)
        elif language == "java":
            r = _run_java_patch(patch, checkout_dir, source_file_rel)
        else:
            r = PatchResult(patch_code=patch, passed=False, error=f"Unsupported language: {language}")
        results.append(r)
    return results


def build_evaluation(
    program_name: str,
    language: str,
    generator_model: str,
    repairer_model: str,
    category: str,
    patch_results: list[PatchResult],
    scenario: str = "llm",
) -> BugEvaluation:
    """Build a BugEvaluation with pass@k computed from patch results."""
    n = len(patch_results)
    c = sum(1 for r in patch_results if r.passed)
    pak: dict[int, float] = {}
    for k in config.PASS_K_VALUES:
        if n >= k:
            pak[k] = pass_at_k(n, c, k)
    return BugEvaluation(
        program_name=program_name,
        language=language,
        generator_model=generator_model,
        repairer_model=repairer_model,
        category=category,
        n_samples=n,
        n_correct=c,
        pass_at_k=pak,
        patch_results=patch_results,
        scenario=scenario,
    )


# ---------------------------------------------------------------------------
# Result aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_matrix(evaluations: list[BugEvaluation]) -> dict:
    """Build the 7x7 cross-evaluation matrix from individual evaluations.

    Returns a nested dict:  matrix[generator][repairer] = {k: avg_pass_at_k}
    """
    from collections import defaultdict
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        groups[(ev.generator_model, ev.repairer_model)].append(
            (ev.n_samples, ev.n_correct)
        )
    matrix: dict[str, dict[str, dict[int, float]]] = {}
    for (gen, rep), nc_pairs in groups.items():
        if gen not in matrix:
            matrix[gen] = {}
        matrix[gen][rep] = compute_pass_at_k_batch(nc_pairs)
    return matrix


def save_evaluation(evaluation: BugEvaluation, out_dir: str | None = None) -> str:
    """Persist a single evaluation to JSON."""
    if out_dir is None:
        out_dir = config.EVAL_DIR
    subdir = os.path.join(
        out_dir, evaluation.generator_model, evaluation.repairer_model
    )
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{evaluation.program_name}.json")
    data = {
        "program_name": evaluation.program_name,
        "language": evaluation.language,
        "generator_model": evaluation.generator_model,
        "repairer_model": evaluation.repairer_model,
        "category": evaluation.category,
        "scenario": evaluation.scenario,
        "n_samples": evaluation.n_samples,
        "n_correct": evaluation.n_correct,
        "pass_at_k": {str(k): v for k, v in evaluation.pass_at_k.items()},
        "patches": [
            {"passed": pr.passed, "error": pr.error, "output_snippet": pr.test_output[:500]}
            for pr in evaluation.patch_results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_matrix(matrix: dict, k: int = 1) -> str:
    """Generate a LaTeX table for the cross-evaluation matrix at a given k."""
    models = config.MODEL_ORDER
    labels = {m: config.MODELS[m].display_name for m in models}

    header = " & ".join(["Generator $\\downarrow$ / Repairer $\\rightarrow$"] + [labels[m] for m in models])
    rows: list[str] = []
    for gen in models:
        cells = [labels[gen]]
        for rep in models:
            val = matrix.get(gen, {}).get(rep, {}).get(k, None)
            if val is not None:
                cells.append(f"{val:.1%}")
            else:
                cells.append("--")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Cross-Model Repair Matrix (pass@{k})}}
    \\label{{tab:matrix_pass_at_{k}}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{l{'c' * len(models)}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
    \\end{{tabular}}}}
    \\end{{table}}
    """)


def generate_latex_taxonomy(evaluations: list[BugEvaluation], k: int = 1) -> str:
    """LaTeX table breaking down pass@k by taxonomy category."""
    from collections import defaultdict

    models = config.MODEL_ORDER
    labels = {m: config.MODELS[m].display_name for m in models}
    categories = list(config.TAXONOMY.keys())

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        groups[(ev.repairer_model, ev.category)].append((ev.n_samples, ev.n_correct))

    header = " & ".join(["Category"] + [labels[m] for m in models])
    rows: list[str] = []
    for cat in categories:
        cells = [config.TAXONOMY[cat]]
        for model in models:
            nc = groups.get((model, cat), [])
            if nc:
                val = compute_pass_at_k_batch(nc, [k]).get(k, 0.0)
                cells.append(f"{val:.1%}")
            else:
                cells.append("--")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Repair Success by Bug Category (pass@{k})}}
    \\label{{tab:taxonomy_pass_at_{k}}}
    \\begin{{tabular}}{{l{'c' * len(models)}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\end{{table}}
    """)  # end generate_latex_taxonomy


SCENARIO_LABELS: dict[str, str] = {
    "human": "Human-Only Bugs",
    "llm": "LLM-Only Bugs",
    "compound": "Compound (Human+LLM)",
}


def generate_latex_scenario(evaluations: list[BugEvaluation], k: int = 1) -> str:
    """LaTeX table comparing pass@k across the three evaluation scenarios."""
    from collections import defaultdict

    models = config.MODEL_ORDER
    labels = {m: config.MODELS[m].display_name for m in models}
    scenarios = ["human", "llm", "compound"]

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        groups[(ev.repairer_model, ev.scenario)].append((ev.n_samples, ev.n_correct))

    header = " & ".join(["Scenario"] + [labels[m] for m in models])
    rows: list[str] = []
    for scenario in scenarios:
        cells = [SCENARIO_LABELS.get(scenario, scenario)]
        for model in models:
            nc = groups.get((model, scenario), [])
            if nc:
                val = compute_pass_at_k_batch(nc, [k]).get(k, 0.0)
                cells.append(f"{val:.1%}")
            else:
                cells.append("--")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Repair Success by Evaluation Scenario (pass@{k})}}
    \\label{{tab:scenario_pass_at_{k}}}
    \\begin{{tabular}}{{l{'c' * len(models)}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\end{{table}}
    """)  # end generate_latex_scenario


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

@dataclass
class PairwiseComparison:
    model_a: str
    model_b: str
    mcnemar_chi2: float
    mcnemar_p: float
    cliffs_delta: float
    effect_size_label: str  # "negligible", "small", "medium", "large"
    n_both_fix: int
    n_a_only: int
    n_b_only: int
    n_neither: int


def bootstrap_pass_at_k_ci(
    results: list[tuple[int, int]],
    k: int = 1,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for pass@k.

    *results* is a list of (n_samples, n_correct) per program.
    Returns (mean, ci_lower, ci_upper).
    """
    if not results:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(rng_seed)
    scores = np.array([pass_at_k(n, c, k) for n, c in results if n >= k])
    if len(scores) == 0:
        return (0.0, 0.0, 0.0)

    mean = float(np.mean(scores))
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (mean, ci_lower, ci_upper)


def mcnemar_test(
    evaluations: list[BugEvaluation],
    model_a: str,
    model_b: str,
) -> PairwiseComparison:
    """McNemar's test for paired model comparison on the same bug set.

    Builds the 2x2 contingency table from evaluations where both models
    attempted the same program (matched by program_name + generator_model + category).
    """
    a_results: dict[tuple, bool] = {}
    b_results: dict[tuple, bool] = {}

    for ev in evaluations:
        key = (ev.program_name, ev.generator_model, ev.category)
        fixed = ev.n_correct > 0
        if ev.repairer_model == model_a:
            a_results[key] = fixed
        elif ev.repairer_model == model_b:
            b_results[key] = fixed

    shared_keys = set(a_results.keys()) & set(b_results.keys())
    n_both = n_a_only = n_b_only = n_neither = 0

    for key in shared_keys:
        a_fixed = a_results[key]
        b_fixed = b_results[key]
        if a_fixed and b_fixed:
            n_both += 1
        elif a_fixed and not b_fixed:
            n_a_only += 1
        elif not a_fixed and b_fixed:
            n_b_only += 1
        else:
            n_neither += 1

    discordant = n_a_only + n_b_only
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / discordant
        from scipy import stats as sp_stats
        p_value = float(sp_stats.chi2.sf(chi2, df=1))

    delta = _cliffs_delta(a_results, b_results, shared_keys)

    return PairwiseComparison(
        model_a=model_a, model_b=model_b,
        mcnemar_chi2=chi2, mcnemar_p=p_value,
        cliffs_delta=delta, effect_size_label=_cliffs_delta_label(delta),
        n_both_fix=n_both, n_a_only=n_a_only,
        n_b_only=n_b_only, n_neither=n_neither,
    )


def _cliffs_delta(
    a_results: dict, b_results: dict, shared_keys: set
) -> float:
    """Cliff's delta for two matched sets of binary outcomes."""
    if not shared_keys:
        return 0.0
    dominance = sum(
        (1 if int(a_results[k]) > int(b_results[k])
         else -1 if int(a_results[k]) < int(b_results[k])
         else 0)
        for k in shared_keys
    )
    return dominance / len(shared_keys)


def _cliffs_delta_label(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    return "large"


def wilcoxon_test_by_category(
    evaluations: list[BugEvaluation],
    model_a: str,
    model_b: str,
) -> dict[str, float]:
    """Wilcoxon signed-rank test comparing repair rates by category.

    Returns {category: p_value}.
    """
    from scipy import stats as sp_stats

    categories = list(config.TAXONOMY.keys())
    results: dict[str, float] = {}

    for cat in categories:
        a_by_prog: dict[str, list[float]] = defaultdict(list)
        b_by_prog: dict[str, list[float]] = defaultdict(list)

        for ev in evaluations:
            if ev.category != cat:
                continue
            rate = ev.n_correct / max(ev.n_samples, 1)
            if ev.repairer_model == model_a:
                a_by_prog[ev.program_name].append(rate)
            elif ev.repairer_model == model_b:
                b_by_prog[ev.program_name].append(rate)

        shared = set(a_by_prog.keys()) & set(b_by_prog.keys())
        a_rates = [float(np.mean(a_by_prog[p])) for p in shared]
        b_rates = [float(np.mean(b_by_prog[p])) for p in shared]

        if len(a_rates) < 5:
            results[cat] = float("nan")
            continue

        try:
            _, p = sp_stats.wilcoxon(a_rates, b_rates, zero_method="pratt")
            results[cat] = float(p)
        except ValueError:
            results[cat] = float("nan")

    return results


def compute_all_pairwise(
    evaluations: list[BugEvaluation],
    models: list[str] | None = None,
) -> list[PairwiseComparison]:
    """Compute McNemar's test for all model pairs."""
    if models is None:
        models = [m for m in config.MODEL_ORDER
                  if any(ev.repairer_model == m for ev in evaluations)]

    comparisons: list[PairwiseComparison] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            comparisons.append(mcnemar_test(evaluations, a, b))
    return comparisons


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def generate_latex_significance_matrix(
    comparisons: list[PairwiseComparison],
) -> str:
    """LaTeX table of pairwise p-values with significance markers."""
    models_in = set()
    for c in comparisons:
        models_in.add(c.model_a)
        models_in.add(c.model_b)
    models = [m for m in config.MODEL_ORDER if m in models_in]
    labels = {m: config.MODELS[m].display_name for m in models}

    p_lookup: dict[tuple[str, str], float] = {}
    for c in comparisons:
        p_lookup[(c.model_a, c.model_b)] = c.mcnemar_p
        p_lookup[(c.model_b, c.model_a)] = c.mcnemar_p

    header = " & ".join([""] + [labels[m] for m in models])
    rows: list[str] = []
    for a in models:
        cells = [labels[a]]
        for b in models:
            if a == b:
                cells.append("--")
            else:
                p = p_lookup.get((a, b), float("nan"))
                if np.isnan(p):
                    cells.append("--")
                else:
                    cells.append(f"{p:.3f}{significance_marker(p)}")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Pairwise McNemar's Test ($*\\ p<0.05$, $**\\ p<0.01$, $***\\ p<0.001$)}}
    \\label{{tab:significance}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{l{'c' * len(models)}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
    \\end{{tabular}}}}
    \\end{{table}}
    """)


def generate_latex_matrix_with_ci(
    evaluations: list[BugEvaluation],
    k: int = 1,
    n_bootstrap: int = 10_000,
) -> str:
    """Cross-evaluation matrix with bootstrap 95% CIs."""
    models = config.MODEL_ORDER
    labels = {m: config.MODELS[m].display_name for m in models}

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        groups[(ev.generator_model, ev.repairer_model)].append(
            (ev.n_samples, ev.n_correct)
        )

    header = " & ".join(
        ["Gen. $\\downarrow$ / Rep. $\\rightarrow$"]
        + [labels[m] for m in models]
    )
    rows: list[str] = []
    for gen in models:
        cells = [labels[gen]]
        for rep in models:
            results = groups.get((gen, rep), [])
            if results:
                mean, lo, hi = bootstrap_pass_at_k_ci(results, k=k, n_bootstrap=n_bootstrap)
                cells.append(f"{mean:.1%} [{lo:.1%}, {hi:.1%}]")
            else:
                cells.append("--")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Cross-Model Repair Matrix with 95\\% Bootstrap CI (pass@{k})}}
    \\label{{tab:matrix_ci_pass_at_{k}}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{l{'c' * len(models)}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
    \\end{{tabular}}}}
    \\end{{table}}
    """)
