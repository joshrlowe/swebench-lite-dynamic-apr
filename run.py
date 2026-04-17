import logging, time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from models import get_client, ALL_MODEL_NAMES, N_REPAIR_SAMPLES_DEFAULT, load_project_dotenv, load_swebench_cases
from pipeline import ( SWEBenchRunner, inject_bug, human_written_bug_cases, llm_injected_bug_cases, compound_bug_cases, run_swebench_repair_loop, )

load_project_dotenv(Path(__file__).resolve().parent)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

N_SAMPLES = N_REPAIR_SAMPLES_DEFAULT
INJECTION_DIR = "results/injections"

# Timestamp for log lines
def _ts():
    return datetime.now(UTC).strftime("%H:%M:%S")

# Repair the human-written bugs across all models
def run_condition_a(base_cases, runner):
    print("Human-Written Bug Repair")

    cases = human_written_bug_cases(base_cases)
    results = []

    for model_name in ALL_MODEL_NAMES:
        print(f"\n[{_ts()}] {model_name}: repairing {len(cases)} human-written bugs")
        model = get_client(model_name)
        t_start = time.monotonic()

        for case in cases:
            try:
                result = run_swebench_repair_loop( case, repairing_model=model, runner=runner, n_samples=N_SAMPLES, )
                results.append((model_name, case, result))
            except Exception as e:
                print(f"[{_ts()}] [{model_name}] {case.display_name}: ERROR {e}", flush=True)

        elapsed = time.monotonic() - t_start
        print(f"[{_ts()}] {model_name} done ({elapsed/60:.1f} min)")
        SWEBenchRunner.docker_prune()

    print("\nHuman-Written Bug Repair Results")
    per_model = defaultdict(lambda: {"resolved": 0, "attempted": 0})
    for model_name, case, result in results:
        per_model[model_name]["resolved"] += result.get("resolved", 0)
        per_model[model_name]["attempted"] += result.get("attempted", 0)
    for model_name in ALL_MODEL_NAMES:
        s = per_model[model_name]
        rate = s["resolved"] / s["attempted"] * 100 if s["attempted"] else 0
        print(f"  {model_name}: {s['resolved']}/{s['attempted']} resolved ({rate:.1f}%)")

    return results

# Have each model inject bugs into the codebase, single or compound
def run_injection(base_cases, runner, compound=False):
    mode = "compound" if compound else "single-fault"
    print(f"\n{'=' * 60}")
    print(f"Injection: {mode}")
    print("=" * 60)

    results = []
    for model_name in ALL_MODEL_NAMES:
        print(f"\n[{_ts()}] {model_name}: injecting {mode} bugs into {len(base_cases)} cases")
        model = get_client(model_name)
        t_start = time.monotonic()
        valid_count = 0

        for case in base_cases:
            bug_type = case.get("assigned_bug_type", "variable_naming")
            try:
                ok = inject_bug(model, case, bug_type, runner=runner, compound=compound, output_dir=INJECTION_DIR)
                results.append((model_name, case, ok))
                if ok:
                    valid_count += 1
            except Exception as e:
                print(f"[{_ts()}] [{model_name}] {case['instance_id']}: ERROR {e}", flush=True)

        elapsed = time.monotonic() - t_start
        print(f"[{_ts()}] {model_name}: {valid_count}/{len(base_cases)} valid ({elapsed/60:.1f} min)")
        SWEBenchRunner.docker_prune()

    print(f"\n- {mode.title()} Injection Results -")
    per_model = defaultdict(lambda: {"valid": 0, "total": 0})
    for model_name, case, ok in results:
        per_model[model_name]["total"] += 1
        if ok:
            per_model[model_name]["valid"] += 1
    for model_name in ALL_MODEL_NAMES:
        s = per_model[model_name]
        rate = s["valid"] / s["total"] * 100 if s["total"] else 0
        print(f"  {model_name}: {s['valid']}/{s['total']} valid ({rate:.1f}%)")

    return results

# Repair LLM-injected or compound bugs and print the cross-model matrix
def run_condition_bc(base_cases, runner, compound=False):
    print("Compound Bug Repair" if compound else "LLM-Injected Bug Repair")

    repair_cases = compound_bug_cases(INJECTION_DIR, base_cases) if compound else llm_injected_bug_cases(INJECTION_DIR, base_cases)
        
    if not repair_cases:
        print("  No valid injections found - skipping")
        return []

    results = []
    for model_name in ALL_MODEL_NAMES:
        print(f"\n[{_ts()}] {model_name}: repairing {len(repair_cases)} injected bugs")
        model = get_client(model_name)
        t_start = time.monotonic()

        for case in repair_cases:
            try:
                result = run_swebench_repair_loop( case, repairing_model=model, runner=runner, n_samples=N_SAMPLES, )
                results.append((model_name, case, result))
            except Exception as e:
                print(f"[{_ts()}] [{model_name}] {case.display_name}: ERROR {e}", flush=True)

        elapsed = time.monotonic() - t_start
        print(f"[{_ts()}] {model_name} done ({elapsed/60:.1f} min)")
        SWEBenchRunner.docker_prune()

    print(f"\n- {label.split(':')[1].strip()} Results -")
    per_model = defaultdict(lambda: {"resolved": 0, "attempted": 0})
    injectors_seen = set()
    matrix = defaultdict(lambda: {"resolved": 0, "attempted": 0})

    for model_name, case, result in results:
        per_model[model_name]["resolved"] += result.get("resolved", 0)
        per_model[model_name]["attempted"] += result.get("attempted", 0)
        injector = getattr(case, "source_model_name", None)
        if injector:
            injectors_seen.add(injector)
            matrix[(injector, model_name)]["attempted"] += result.get("attempted", 0)
            matrix[(injector, model_name)]["resolved"] += result.get("resolved", 0)

    for model_name in ALL_MODEL_NAMES:
        s = per_model[model_name]
        rate = s["resolved"] / s["attempted"] * 100 if s["attempted"] else 0
        print(f"  {model_name}: {s['resolved']}/{s['attempted']} resolved ({rate:.1f}%)")

    if len(injectors_seen) > 1:
        print(f"\n  Injector x Repairer matrix (resolved/attempted):")
        header = "  " + " " * 24 + "".join(f"{m[:12]:>14}" for m in ALL_MODEL_NAMES)
        print(header)
        for inj in sorted(injectors_seen):
            row = f"  {inj:<24}"
            for rep in ALL_MODEL_NAMES:
                cell = matrix.get((inj, rep), {"resolved": 0, "attempted": 0})
                row += f"{cell['resolved']}/{cell['attempted']:>13}" if cell["attempted"] > 0 else f"{'-':>14}"
            print(row)

    return results

# Orchestrator that kicks off all experiment conditions
def main():
    print("Loading SWE-bench Lite cases...")
    base_cases = load_swebench_cases()
    runner = SWEBenchRunner(timeout=600, cache_level="env", clean=True)

    t_start = time.monotonic()

    run_condition_a(base_cases, runner)

    run_injection(base_cases, runner, compound=False)
    run_condition_bc(base_cases, runner, compound=False)

    run_injection(base_cases, runner, compound=True)
    run_condition_bc(base_cases, runner, compound=True)

    total_elapsed = time.monotonic() - t_start
    print("All Experiments Complete")

if __name__ == "__main__":
    main()
