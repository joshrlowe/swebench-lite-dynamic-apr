"""LLM-Adversarial APR Benchmark — Orchestration CLI.

Usage:
    python main.py baseline           — Phase 0: Evaluate on original human bugs
    python main.py generate           — Phase 1: Generate adversarial (LLM-only) bugs
    python main.py generate-compound  — Phase 1b: Generate compound (human+LLM) bugs
    python main.py repair             — Phase 2: Run 7×7 cross-model repair on LLM bugs
    python main.py repair-compound    — Phase 2b: Run repair matrix on compound bugs
    python main.py evaluate           — Phase 3: Compute pass@k and generate LaTeX tables
    python main.py all                — Run all phases end-to-end
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import config
from api_manager import APIManager
from bug_generator import (
    GeneratedBug,
    generate_bug,
    generate_compound_bug,
    load_all_bugs,
    load_all_compound_bugs,
    save_bug,
    save_compound_bug,
)
from dataset_loader import BenchmarkCase, load_all, load_quixbugs
from evaluator import (
    BugEvaluation,
    aggregate_matrix,
    compute_all_pairwise,
    generate_latex_matrix,
    generate_latex_matrix_with_ci,
    generate_latex_scenario,
    generate_latex_significance_matrix,
    generate_latex_taxonomy,
)
from repair_engine import repair_baseline_bug, run_repair_matrix

logger = logging.getLogger("apr")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logfile = RotatingFileHandler(
        os.path.join(config.RESULTS_DIR, "experiment.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(logfile)


# ---------------------------------------------------------------------------
# Phase 0: Baseline — repair original human-induced bugs
# ---------------------------------------------------------------------------

async def phase_baseline(
    api: APIManager,
    benchmark: str = "quixbugs",
) -> list[BugEvaluation]:
    """Evaluate all models on repairing the original human-induced bugs."""
    if benchmark == "quixbugs":
        cases = load_quixbugs()
    else:
        cases = load_all()

    cases_with_bugs = [c for c in cases if c.buggy_code.strip()]
    available = api.get_available_models()
    evaluations: list[BugEvaluation] = []

    total = len(cases_with_bugs) * len(available)
    done = 0

    logger.info(
        "Phase 0 (Baseline): %d programs × %d models = %d tasks",
        len(cases_with_bugs), len(available), total,
    )

    for case in cases_with_bugs:
        for model_key in available:
            done += 1
            eval_path = os.path.join(
                config.BASELINE_DIR, "human", model_key, f"{case.name}.json"
            )
            if os.path.exists(eval_path):
                logger.debug("Skipping existing baseline eval: %s", eval_path)
                continue

            logger.info("[%d/%d] baseline: %s / %s", done, total, case.name, model_key)
            try:
                ev = await repair_baseline_bug(api, case, model_key)
                evaluations.append(ev)
            except Exception as exc:
                logger.error("Baseline repair failed for %s/%s: %s", case.name, model_key, exc)

    api.flush_costs()
    logger.info("Phase 0 complete: %d baseline evaluations", len(evaluations))
    return evaluations


# ---------------------------------------------------------------------------
# Phase 1: Generate adversarial bugs
# ---------------------------------------------------------------------------

async def phase_generate(
    api: APIManager,
    benchmark: str = "quixbugs",
) -> list[GeneratedBug]:
    """Generate taxonomy-guided adversarial bugs for all models × categories."""
    if benchmark == "quixbugs":
        cases = load_quixbugs()
    else:
        cases = load_all()

    available = api.get_available_models()
    categories = list(config.TAXONOMY.keys())
    all_bugs: list[GeneratedBug] = []

    total = len(cases) * len(available) * len(categories)
    done = 0

    logger.info(
        "Phase 1: Generating bugs — %d programs × %d models × %d categories = %d tasks",
        len(cases), len(available), len(categories), total,
    )

    for case in cases:
        for model_key in available:
            for category in categories:
                done += 1
                bug_path = os.path.join(
                    config.BUGS_DIR, model_key, category, f"{case.name}.json"
                )
                if os.path.exists(bug_path):
                    logger.debug("Skipping existing bug: %s", bug_path)
                    continue

                logger.info("[%d/%d] %s / %s / %s", done, total, case.name, model_key, category)
                try:
                    bug = await generate_bug(api, case, model_key, category)
                    if bug and bug.validated:
                        save_bug(bug)
                        all_bugs.append(bug)
                    else:
                        logger.info("No valid bug generated — skipping")
                except Exception as exc:
                    logger.error("Bug generation failed: %s", exc)

    api.flush_costs()
    logger.info("Phase 1 complete: %d new validated bugs", len(all_bugs))
    return all_bugs


# ---------------------------------------------------------------------------
# Phase 1b: Generate compound bugs (human + LLM faults)
# ---------------------------------------------------------------------------

async def phase_generate_compound(
    api: APIManager,
    benchmark: str = "quixbugs",
) -> list[GeneratedBug]:
    """Generate compound bugs: inject additional LLM faults into human-buggy code."""
    if benchmark == "quixbugs":
        cases = load_quixbugs()
    else:
        cases = load_all()

    cases_with_bugs = [c for c in cases if c.buggy_code.strip()]
    available = api.get_available_models()
    categories = list(config.TAXONOMY.keys())
    all_bugs: list[GeneratedBug] = []

    total = len(cases_with_bugs) * len(available) * len(categories)
    done = 0

    logger.info(
        "Phase 1b: Generating compound bugs — %d programs × %d models × %d categories = %d tasks",
        len(cases_with_bugs), len(available), len(categories), total,
    )

    for case in cases_with_bugs:
        for model_key in available:
            for category in categories:
                done += 1
                bug_path = os.path.join(
                    config.COMPOUND_BUGS_DIR, model_key, category, f"{case.name}.json"
                )
                if os.path.exists(bug_path):
                    logger.debug("Skipping existing compound bug: %s", bug_path)
                    continue

                logger.info(
                    "[%d/%d] compound: %s / %s / %s",
                    done, total, case.name, model_key, category,
                )
                try:
                    bug = await generate_compound_bug(api, case, model_key, category)
                    if bug and bug.validated:
                        save_compound_bug(bug)
                        all_bugs.append(bug)
                    else:
                        logger.info("No valid compound bug generated — skipping")
                except Exception as exc:
                    logger.error("Compound bug generation failed: %s", exc)

    api.flush_costs()
    logger.info("Phase 1b complete: %d new validated compound bugs", len(all_bugs))
    return all_bugs


# ---------------------------------------------------------------------------
# Phase 2b: Cross-model repair for compound bugs
# ---------------------------------------------------------------------------

async def phase_repair_compound(api: APIManager) -> list[BugEvaluation]:
    """Load all compound bugs and run the cross-model repair matrix."""
    bugs = load_all_compound_bugs()
    if not bugs:
        logger.warning("No compound bugs found — run 'generate-compound' phase first")
        return []

    logger.info("Phase 2b: Repairing %d compound bugs across the model matrix", len(bugs))
    evaluations = await run_repair_matrix(
        api, bugs, scenario="compound", eval_dir=config.COMPOUND_EVAL_DIR,
    )
    logger.info("Phase 2b complete: %d compound evaluations", len(evaluations))
    return evaluations


# ---------------------------------------------------------------------------
# Phase 2: Cross-model repair
# ---------------------------------------------------------------------------

async def phase_repair(api: APIManager) -> list[BugEvaluation]:
    """Load all validated bugs and run the 7×7 repair matrix."""
    bugs = load_all_bugs()
    if not bugs:
        logger.warning("No validated bugs found — run 'generate' phase first")
        return []

    logger.info("Phase 2: Repairing %d bugs across the model matrix", len(bugs))
    evaluations = await run_repair_matrix(api, bugs)
    logger.info("Phase 2 complete: %d evaluations", len(evaluations))
    return evaluations


# ---------------------------------------------------------------------------
# Phase 3: Evaluate and produce tables
# ---------------------------------------------------------------------------

def _load_evaluations_from_dir(eval_dir: str, default_scenario: str) -> list[BugEvaluation]:
    """Load all BugEvaluation JSONs from a nested directory tree."""
    import json

    evaluations: list[BugEvaluation] = []
    if not os.path.isdir(eval_dir):
        return evaluations

    for gen_dir in os.listdir(eval_dir):
        gen_path = os.path.join(eval_dir, gen_dir)
        if not os.path.isdir(gen_path):
            continue
        for rep_dir in os.listdir(gen_path):
            rep_path = os.path.join(gen_path, rep_dir)
            if not os.path.isdir(rep_path):
                continue
            for fname in os.listdir(rep_path):
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(rep_path, fname)) as f:
                        data = json.load(f)
                    ev = BugEvaluation(
                        program_name=data["program_name"],
                        language=data["language"],
                        generator_model=data["generator_model"],
                        repairer_model=data["repairer_model"],
                        category=data["category"],
                        n_samples=data["n_samples"],
                        n_correct=data["n_correct"],
                        pass_at_k={int(k): v for k, v in data["pass_at_k"].items()},
                        scenario=data.get("scenario", default_scenario),
                    )
                    evaluations.append(ev)
                except Exception as exc:
                    logger.error("Failed loading eval %s: %s", fname, exc)
    return evaluations


def phase_evaluate() -> None:
    """Load all evaluations from disk and produce LaTeX tables + figures."""
    baseline_evals = _load_evaluations_from_dir(config.BASELINE_DIR, "human")
    llm_evals = _load_evaluations_from_dir(config.EVAL_DIR, "llm")
    compound_evals = _load_evaluations_from_dir(config.COMPOUND_EVAL_DIR, "compound")

    all_evaluations = baseline_evals + llm_evals + compound_evals

    if not all_evaluations:
        logger.warning("No evaluations loaded from any scenario")
        return

    logger.info(
        "Loaded %d evaluations (baseline=%d, llm=%d, compound=%d) — generating tables",
        len(all_evaluations), len(baseline_evals), len(llm_evals), len(compound_evals),
    )

    os.makedirs(config.TABLES_DIR, exist_ok=True)

    llm_matrix = aggregate_matrix(llm_evals) if llm_evals else {}
    for k in config.PASS_K_VALUES:
        if llm_matrix:
            table = generate_latex_matrix(llm_matrix, k=k)
            path = os.path.join(config.TABLES_DIR, f"matrix_pass_at_{k}.tex")
            with open(path, "w") as f:
                f.write(table)
            logger.info("Wrote %s", path)

        if llm_evals:
            table = generate_latex_taxonomy(llm_evals, k=k)
            path = os.path.join(config.TABLES_DIR, f"taxonomy_pass_at_{k}.tex")
            with open(path, "w") as f:
                f.write(table)
            logger.info("Wrote %s", path)

        if all_evaluations:
            table = generate_latex_scenario(all_evaluations, k=k)
            path = os.path.join(config.TABLES_DIR, f"scenario_pass_at_{k}.tex")
            with open(path, "w") as f:
                f.write(table)
            logger.info("Wrote %s", path)

    if llm_matrix:
        _generate_figures(llm_matrix, llm_evals)

    # Statistical significance tables
    if llm_evals:
        comparisons = compute_all_pairwise(llm_evals)
        if comparisons:
            table = generate_latex_significance_matrix(comparisons)
            path = os.path.join(config.TABLES_DIR, "significance_matrix.tex")
            with open(path, "w") as f:
                f.write(table)
            logger.info("Wrote %s", path)

        for k in config.PASS_K_VALUES:
            table = generate_latex_matrix_with_ci(llm_evals, k=k)
            path = os.path.join(config.TABLES_DIR, f"matrix_ci_pass_at_{k}.tex")
            with open(path, "w") as f:
                f.write(table)
            logger.info("Wrote %s", path)

    _print_summary(all_evaluations, llm_matrix)


def _generate_figures(matrix: dict, evaluations: list[BugEvaluation]) -> None:
    """Generate publication-quality figures using seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping figures")
        return

    from evaluator import bootstrap_pass_at_k_ci
    from collections import defaultdict

    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Publication style
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.grid": False,
        "figure.dpi": 300,
    })

    PALETTE = sns.color_palette("muted", n_colors=7)
    models = [m for m in config.MODEL_ORDER if m in matrix]
    if not models:
        return
    labels = [config.MODELS[m].display_name for m in models]
    model_colors = {m: PALETTE[i] for i, m in enumerate(models)}

    def _save(fig, name):
        for ext in ["pdf", "png"]:
            p = os.path.join(config.FIGURES_DIR, f"{name}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote figures/%s.{pdf,png}", name)

    # --- 1. Cross-model repair heatmap ---
    for k in config.PASS_K_VALUES:
        data = np.array([
            [matrix.get(g, {}).get(r, {}).get(k, 0.0) * 100 for r in models]
            for g in models
        ])
        fig, ax = plt.subplots(figsize=(7, 6))
        mean_val = np.mean(data[data > 0]) if np.any(data > 0) else 50
        sns.heatmap(
            data, annot=True, fmt=".1f", cmap="RdYlGn",
            center=mean_val, vmin=0, vmax=100,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="gray", ax=ax,
        )
        # Bold diagonal
        for i in range(len(models)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2.5))
        ax.set_xlabel("Repairer Model")
        ax.set_ylabel("Generator Model")
        ax.set_title(f"Cross-Model Repair Matrix — pass@{k} (%)")
        ax.tick_params(axis="x", rotation=45)
        _save(fig, f"matrix_pass_at_{k}")

    # --- 2. Self-repair bias bar chart ---
    self_scores, cross_scores, bias = [], [], []
    for m in models:
        s = matrix.get(m, {}).get(m, {}).get(1, 0.0) * 100
        cross_vals = [matrix.get(m, {}).get(r, {}).get(1, 0.0) * 100
                      for r in models if r != m]
        c = np.mean(cross_vals) if cross_vals else 0.0
        self_scores.append(s)
        cross_scores.append(c)
        bias.append(s - c)

    # Sort by bias magnitude
    order = np.argsort(bias)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, [self_scores[i] for i in order], w,
           label="Self-Repair", color=PALETTE[0])
    ax.bar(x + w / 2, [cross_scores[i] for i in order], w,
           label="Cross-Repair (avg)", color=PALETTE[1])
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Self-Repair Bias: Self vs. Cross-Model Repair")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    _save(fig, "self_vs_cross_repair")

    # --- 3. Per-bug-category grouped bar chart ---
    categories = list(config.TAXONOMY.keys())
    cat_labels = list(config.TAXONOMY.values())

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        groups[(ev.repairer_model, ev.category)].append((ev.n_samples, ev.n_correct))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_cats = len(categories)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_cats)

    for mi, m in enumerate(models):
        vals = []
        for cat in categories:
            nc = groups.get((m, cat), [])
            if nc:
                from evaluator import compute_pass_at_k_batch
                vals.append(compute_pass_at_k_batch(nc, [1]).get(1, 0.0) * 100)
            else:
                vals.append(0.0)
        offset = (mi - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.9, label=labels[mi], color=model_colors[m])

    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Repair Success by Bug Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=20, ha="right")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    ax.set_ylim(0, 100)
    _save(fig, "category_breakdown")

    # --- 4. Difficulty progression line chart ---
    scenarios = ["human", "llm", "compound"]
    scenario_labels = ["Human", "LLM", "Compound"]

    scenario_groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for ev in evaluations:
        scenario_groups[(ev.repairer_model, ev.scenario)].append(
            (ev.n_samples, ev.n_correct)
        )

    fig, ax = plt.subplots(figsize=(5, 4))
    for mi, m in enumerate(models):
        vals = []
        for s in scenarios:
            nc = scenario_groups.get((m, s), [])
            if nc:
                from evaluator import compute_pass_at_k_batch
                vals.append(compute_pass_at_k_batch(nc, [1]).get(1, 0.0) * 100)
            else:
                vals.append(float("nan"))
        ax.plot(scenario_labels, vals, marker="o", label=labels[mi],
                color=model_colors[m], linewidth=1.5, markersize=5)

    ax.set_ylabel("Mean pass@1 (%)")
    ax.set_title("Repair Difficulty by Bug Origin")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 100)
    _save(fig, "difficulty_progression")

    # --- 5. Box plots for repair rate distributions ---
    fig, axes = plt.subplots(1, min(3, len(set(ev.scenario for ev in evaluations))),
                             figsize=(7, 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    scenario_map = {"human": "Human Bugs", "llm": "LLM Bugs", "compound": "Compound Bugs"}
    for ax_i, (scenario, title) in enumerate(scenario_map.items()):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]
        box_data = []
        box_labels_list = []
        for m in models:
            rates = [
                ev.n_correct / max(ev.n_samples, 1) * 100
                for ev in evaluations
                if ev.repairer_model == m and ev.scenario == scenario
            ]
            if rates:
                box_data.append(rates)
                box_labels_list.append(config.MODELS[m].display_name)

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], PALETTE[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(box_labels_list, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        if ax_i == 0:
            ax.set_ylabel("Repair Rate (%)")

    fig.suptitle("Repair Rate Distributions", fontsize=11)
    fig.tight_layout()
    _save(fig, "repair_distributions")


def _print_summary(evaluations: list[BugEvaluation], matrix: dict) -> None:
    """Print a human-readable summary to the console."""
    from collections import Counter

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info("Total evaluations: %d", len(evaluations))

    by_gen = Counter(ev.generator_model for ev in evaluations)
    by_rep = Counter(ev.repairer_model for ev in evaluations)
    by_cat = Counter(ev.category for ev in evaluations)
    by_scenario = Counter(ev.scenario for ev in evaluations)

    logger.info("By generator: %s", dict(by_gen))
    logger.info("By repairer: %s", dict(by_rep))
    logger.info("By category: %s", dict(by_cat))
    logger.info("By scenario: %s", dict(by_scenario))

    for k in config.PASS_K_VALUES:
        vals = [ev.pass_at_k.get(k, 0.0) for ev in evaluations if k in ev.pass_at_k]
        if vals:
            avg = sum(vals) / len(vals)
            logger.info("Overall pass@%d: %.1f%% (n=%d)", k, avg * 100, len(vals))

    models = [m for m in config.MODEL_ORDER if m in matrix]
    if models:
        logger.info("\nSelf-repair bias analysis:")
        for m in models:
            self_val = matrix.get(m, {}).get(m, {}).get(1, 0.0)
            cross_vals = [
                matrix.get(m, {}).get(r, {}).get(1, 0.0) for r in models if r != m
            ]
            avg_cross = sum(cross_vals) / len(cross_vals) if cross_vals else 0.0
            diff = self_val - avg_cross
            logger.info(
                "  %s: self=%.1f%% cross=%.1f%% delta=%+.1f%%",
                config.MODELS[m].display_name, self_val * 100, avg_cross * 100, diff * 100,
            )


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Token estimates derived from actual prompt analysis (chars/4 heuristic)
_EST_PROMPT_TOKENS = {
    "generate": 345,        # bug generation prompt
    "compound": 389,        # compound injection prompt
    "repair": 285,          # repair prompt
    "baseline": 285,        # baseline repair prompt
}
_EST_COMPLETION_TOKENS = {
    "generate": 150,        # code block only
    "compound": 150,        # code block only
    "repair": 400,          # reasoning + code
    "baseline": 400,        # reasoning + code
}


def phase_estimate(
    phases: str = "all",
    k_override: int | None = None,
    benchmark: str = "quixbugs",
) -> None:
    """Estimate API cost for each phase without making any calls."""
    from dataset_loader import load_quixbugs, load_all
    from bug_generator import load_all_bugs, load_all_compound_bugs

    if benchmark == "quixbugs":
        cases = load_quixbugs()
    else:
        cases = load_all()

    cases_with_bugs = [c for c in cases if c.buggy_code.strip()]
    n_programs = len(cases)
    n_buggy = len(cases_with_bugs)
    n_models = len(config.MODEL_ORDER)
    n_categories = len(config.TAXONOMY)
    k = k_override if k_override is not None else config.NUM_SAMPLES

    # Determine which phases to estimate
    if phases == "all":
        phase_list = ["2", "3", "4"]
    else:
        phase_list = [p.strip() for p in phases.split(",")]

    pricing = APIManager.PRICING_PER_1K

    print()
    print("=" * 80)
    print(f"  COST ESTIMATE  |  {n_programs} programs, {n_models} models, "
          f"{n_categories} categories, k={k}")
    print("=" * 80)

    grand_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    for phase in phase_list:
        if phase == "2":
            _estimate_phase(
                title="Phase 2 — Bug Generation",
                description=f"{n_programs} programs × {n_models} models × {n_categories} categories",
                calls_per_model=n_programs * n_categories,
                n_samples_per_call=1,
                prompt_tokens=_EST_PROMPT_TOKENS["generate"],
                completion_tokens=_EST_COMPLETION_TOKENS["generate"],
                pricing=pricing,
                grand_totals=grand_totals,
            )
        elif phase == "3":
            _estimate_phase(
                title="Phase 3 — Compound Bug Injection",
                description=f"{n_buggy} buggy programs × {n_models} models × {n_categories} categories",
                calls_per_model=n_buggy * n_categories,
                n_samples_per_call=1,
                prompt_tokens=_EST_PROMPT_TOKENS["compound"],
                completion_tokens=_EST_COMPLETION_TOKENS["compound"],
                pricing=pricing,
                grand_totals=grand_totals,
            )
        elif phase == "4":
            # Phase 4: each bug is repaired by each model with k samples
            # Bugs come from Phase 2: n_programs * n_models * n_categories
            n_bugs_phase2 = n_programs * n_models * n_categories
            # Plus compound bugs from Phase 3
            n_bugs_phase3 = n_buggy * n_models * n_categories
            total_bugs = n_bugs_phase2 + n_bugs_phase3

            # Each bug repaired by each model
            _estimate_phase(
                title=f"Phase 4 — Cross-Model Repair (k={k})",
                description=(
                    f"({n_bugs_phase2} LLM bugs + {n_bugs_phase3} compound bugs) "
                    f"× {n_models} repairers × {k} samples"
                ),
                calls_per_model=total_bugs * k,
                n_samples_per_call=1,
                prompt_tokens=_EST_PROMPT_TOKENS["repair"],
                completion_tokens=_EST_COMPLETION_TOKENS["repair"],
                pricing=pricing,
                grand_totals=grand_totals,
            )
        else:
            print(f"\n  Unknown phase: {phase}")

    # Grand total
    print()
    print("-" * 80)
    print(f"  {'GRAND TOTAL':<20} | "
          f"{grand_totals['calls']:>10,} calls | "
          f"{grand_totals['prompt_tokens']:>14,} in | "
          f"{grand_totals['completion_tokens']:>14,} out | "
          f"${grand_totals['cost']:>10,.2f}")
    print("=" * 80)
    print()


def _estimate_phase(
    title: str,
    description: str,
    calls_per_model: int,
    n_samples_per_call: int,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, tuple[float, float]],
    grand_totals: dict,
) -> None:
    """Print a cost table for one phase, broken down by model."""
    print()
    print(f"  {title}")
    print(f"  {description}")
    print()
    print(f"  {'Model':<22} | {'Calls':>10} | {'Est. In Tokens':>14} | "
          f"{'Est. Out Tokens':>15} | {'Est. Cost':>10}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*14}-+-{'-'*15}-+-{'-'*10}")

    phase_total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    for model_key in config.MODEL_ORDER:
        spec = config.MODELS[model_key]
        n_calls = calls_per_model * n_samples_per_call
        total_prompt = n_calls * prompt_tokens
        total_completion = n_calls * completion_tokens

        p_rate, c_rate = pricing.get(model_key, (0.0, 0.0))
        cost = (total_prompt / 1000) * p_rate + (total_completion / 1000) * c_rate

        print(f"  {spec.display_name:<22} | {n_calls:>10,} | {total_prompt:>14,} | "
              f"{total_completion:>15,} | ${cost:>9,.2f}")

        phase_total["calls"] += n_calls
        phase_total["prompt_tokens"] += total_prompt
        phase_total["completion_tokens"] += total_completion
        phase_total["cost"] += cost

    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*14}-+-{'-'*15}-+-{'-'*10}")
    print(f"  {'Phase Total':<22} | {phase_total['calls']:>10,} | "
          f"{phase_total['prompt_tokens']:>14,} | {phase_total['completion_tokens']:>15,} | "
          f"${phase_total['cost']:>9,.2f}")

    for k in grand_totals:
        grand_totals[k] += phase_total[k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-Adversarial APR Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=[
            "baseline", "generate", "generate-compound",
            "repair", "repair-compound", "evaluate", "estimate", "all",
        ],
        help="Phase to run",
    )
    parser.add_argument(
        "--benchmark", default="quixbugs",
        choices=["quixbugs", "all"],
        help="Which benchmarks to use (default: quixbugs)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--phase", type=str, default=None,
        help="Phase(s) to estimate cost for: 2, 3, 4, or 'all' (estimate command only)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override NUM_SAMPLES (k) for cost estimation (estimate command only)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "estimate":
        phase_estimate(
            phases=args.phase or "all",
            k_override=args.k,
            benchmark=args.benchmark,
        )
        return

    logger.info("LLM-Adversarial APR Benchmark")
    logger.info("Available models: %s", config.get_available_models())

    api = APIManager()
    start = time.time()

    if args.command in ("baseline", "all"):
        await phase_baseline(api, benchmark=args.benchmark)

    if args.command in ("generate", "all"):
        await phase_generate(api, benchmark=args.benchmark)

    if args.command in ("generate-compound", "all"):
        await phase_generate_compound(api, benchmark=args.benchmark)

    if args.command in ("repair", "all"):
        await phase_repair(api)

    if args.command in ("repair-compound", "all"):
        await phase_repair_compound(api)

    if args.command in ("evaluate", "all"):
        phase_evaluate()

    elapsed = time.time() - start
    logger.info("Total time: %.1f seconds | Total API cost: $%.4f", elapsed, api.total_cost())
    api.flush_costs()


if __name__ == "__main__":
    asyncio.run(main())
