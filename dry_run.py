#!/usr/bin/env python3
"""Dry run: validate Phases 2, 3, 4 with minimal API spend.

Uses 1 model (DeepSeek-V3.2), 2 programs (gcd, bitcount), 1 category (variable_misuse), k=1.
"""

import asyncio
import json
import logging
import os
import sys
import time

import config
from api_manager import APIManager
from bug_generator import GeneratedBug, generate_bug, generate_compound_bug, save_bug, save_compound_bug
from dataset_loader import load_quixbugs
from repair_engine import repair_bug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dry_run")

MODEL = "deepseek_v32"
PROGRAMS = ["gcd", "bitcount"]
CATEGORY = "variable_misuse"
DRY_RUN_DIR = os.path.join(config.RESULTS_DIR, "dry_run")


async def main():
    api = APIManager(concurrency=2)
    start = time.time()

    cases = load_quixbugs()
    selected = [c for c in cases if c.name in PROGRAMS]
    logger.info("Selected %d programs: %s", len(selected), [c.name for c in selected])

    # ------------------------------------------------------------------
    # Phase 2: Bug Generation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: Bug Generation")
    logger.info("=" * 60)

    generated_bugs: list[GeneratedBug] = []
    for case in selected:
        logger.info("Generating %s bug for %s with %s", CATEGORY, case.name, MODEL)
        try:
            bug = await generate_bug(api, case, MODEL, CATEGORY)
            if bug and bug.validated:
                save_bug(bug)
                generated_bugs.append(bug)
                logger.info("  -> VALIDATED: buggy code fails tests")
            else:
                logger.warning("  -> FAILED validation")
        except Exception as exc:
            logger.error("  -> ERROR: %s", exc)

    logger.info("Phase 2 result: %d/%d bugs generated and validated",
                len(generated_bugs), len(selected))

    # ------------------------------------------------------------------
    # Phase 3: Compound Bug Injection
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: Compound Bug Injection")
    logger.info("=" * 60)

    compound_bugs: list[GeneratedBug] = []
    cases_with_bugs = [c for c in selected if c.buggy_code.strip()]
    for case in cases_with_bugs:
        logger.info("Injecting compound %s bug into %s with %s", CATEGORY, case.name, MODEL)
        try:
            bug = await generate_compound_bug(api, case, MODEL, CATEGORY)
            if bug and bug.validated:
                save_compound_bug(bug)
                compound_bugs.append(bug)
                logger.info("  -> VALIDATED: compound code fails tests")
            else:
                logger.warning("  -> FAILED validation")
        except Exception as exc:
            logger.error("  -> ERROR: %s", exc)

    logger.info("Phase 3 result: %d/%d compound bugs generated and validated",
                len(compound_bugs), len(cases_with_bugs))

    # ------------------------------------------------------------------
    # Phase 4: Repair (self-repair only, k=1)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 4: Self-Repair (k=1)")
    logger.info("=" * 60)

    # Temporarily override NUM_SAMPLES for k=1
    original_k = config.NUM_SAMPLES
    config.NUM_SAMPLES = 1

    eval_dir = os.path.join(DRY_RUN_DIR, "evaluations")

    all_bugs = generated_bugs + compound_bugs
    for bug in all_bugs:
        scenario = "llm" if bug not in compound_bugs else "compound"
        logger.info("Repairing %s (%s) with %s", bug.program_name, scenario, MODEL)
        try:
            ev = await repair_bug(api, bug, MODEL, scenario=scenario, eval_dir=eval_dir)
            logger.info("  -> pass@1=%.2f (%d/%d correct)",
                        ev.pass_at_k.get(1, 0.0), ev.n_correct, ev.n_samples)
        except Exception as exc:
            logger.error("  -> ERROR: %s", exc)

    config.NUM_SAMPLES = original_k

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    cost = api.total_cost()
    api.flush_costs()

    logger.info("=" * 60)
    logger.info("DRY RUN COMPLETE")
    logger.info("=" * 60)
    logger.info("Time: %.1f seconds", elapsed)
    logger.info("API cost: $%.4f", cost)
    logger.info("Generated bugs: %d/%d", len(generated_bugs), len(selected))
    logger.info("Compound bugs: %d/%d", len(compound_bugs), len(cases_with_bugs))
    logger.info("Total repairs attempted: %d", len(all_bugs))

    # Verify output files
    logger.info("\nOutput files:")
    for dirpath, dirnames, filenames in os.walk(DRY_RUN_DIR):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            logger.info("  %s (%d bytes)", os.path.relpath(fpath, DRY_RUN_DIR),
                        os.path.getsize(fpath))

    # Check bugs dir too
    for dirpath, dirnames, filenames in os.walk(config.BUGS_DIR):
        for fname in filenames:
            if any(p in dirpath for p in PROGRAMS):
                continue
            fpath = os.path.join(dirpath, fname)
            if fname.rstrip(".json") in PROGRAMS:
                logger.info("  [bugs] %s", os.path.relpath(fpath, config.RESULTS_DIR))


if __name__ == "__main__":
    asyncio.run(main())
