# Can LLMs Fix Each Other's & Human Bugs?

An Automated Program Repair (APR) evaluation framework that studies cross-model repair capabilities and adversarial bug generation across 7 frontier LLMs. Graduate research project at UCF.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
cp .env.example .env  # Then edit with your keys
```

### Required API Keys (.env)

```
OPENAI_API_KEY=sk-...          # GPT-5.3-Codex
ANTHROPIC_API_KEY=sk-ant-...   # Claude Sonnet 4.6, Claude Opus 4.6
OPENROUTER_API_KEY=sk-or-...   # DeepSeek-R1, DeepSeek-V3.2
XAI_API_KEY=xai-...            # Grok 4.1
GEMINI_API_KEY=...             # Gemini 1.5 Pro
```

## Running the Experiment

### Cost Estimation (run first!)

```bash
python main.py estimate --phase all          # Full cost estimate
python main.py estimate --phase 4 --k 1      # Phase 4 at k=1 (cheaper)
python main.py estimate --phase 4 --k 10     # Phase 4 at k=10 (default)
```

### Phase 0: Baseline (repair human-induced bugs)
```bash
python main.py baseline                       # ~$0.02, resumable
```

### Phase 1: Bug Generation (LLM-generated adversarial bugs)
```bash
python main.py generate                       # ~$4.50
```

### Phase 1b: Compound Bug Injection (human + LLM bugs)
```bash
python main.py generate-compound              # ~$4.70
```

### Phase 2: Cross-Model Repair Matrix
```bash
python main.py repair                         # $130-$1300 depending on k
python main.py repair-compound                # Same range
```

### Phase 3: Evaluation & Tables
```bash
python main.py evaluate                       # No API cost, generates LaTeX + figures
```

### Full Pipeline
```bash
python main.py all                            # Runs everything sequentially
```

All phases are **resumable** — they skip already-completed evaluations.

## Project Structure

```
config.py            — Model registry (7 LLMs), taxonomy, experiment parameters
api_manager.py       — Unified async API layer (litellm + OpenAI Responses)
dataset_loader.py    — QuixBugs (40 Python) + Defects4J (60 Java) ingestion
bug_generator.py     — Taxonomy-guided adversarial bug generation (4 categories)
repair_engine.py     — Cross-model repair with structured CoT prompts
evaluator.py         — Sandbox execution, pass@k, LaTeX tables, stats tests
main.py              — CLI orchestrator for all phases
dry_run.py           — Minimal validation run (1 model, 2 programs)
benchmarks/quixbugs/ — QuixBugs benchmark programs
results/             — All experiment outputs (gitignored)
tests/               — Unit tests (108 tests)
```

## Models

| Key | Model | Provider |
|-----|-------|----------|
| gpt53_codex | GPT-5.3-Codex | OpenAI |
| claude_sonnet | Claude Sonnet 4.6 | Anthropic |
| claude_opus | Claude Opus 4.6 | Anthropic |
| deepseek_r1 | DeepSeek-R1 | OpenRouter |
| deepseek_v32 | DeepSeek-V3.2 | OpenRouter |
| grok_41 | Grok 4.1 | xAI |
| gemini_31_pro | Gemini 1.5 Pro | Google |

## Bug Taxonomy

- **Variable/Data Misuse** — swapped variables, wrong references
- **Logic/Condition Error** — off-by-one, flipped conditions
- **Loop/Iteration Flaw** — wrong bounds, skipped elements
- **Function Parameter Error** — wrong argument order, off-by-one args

## Output Files

```
results/
├── baseline/          — Phase 0 evaluations (human bugs)
├── bugs/              — Phase 1 generated bugs
├── bugs_compound/     — Phase 1b compound bugs
├── evaluations/       — Phase 2 repair results
├── evaluations_compound/ — Phase 2b compound repair results
├── patches/           — All candidate patches
├── tables/            — LaTeX tables (.tex)
├── figures/           — Publication figures (.pdf, .png)
├── api_costs.jsonl    — Per-call cost tracking
└── experiment.log     — Rotating log (10MB max, 5 backups)
```

## Tests

```bash
python -m pytest tests/ -v --timeout=60
```

108 tests covering: pass@k estimator, QuixBugs test generation, sandbox execution, prompt templates, API manager (mocked), and result I/O.

## Evaluation JSON Schema

Each evaluation file contains:
```json
{
  "program_name": "gcd",
  "language": "python",
  "generator_model": "claude_opus",
  "repairer_model": "deepseek_r1",
  "category": "variable_misuse",
  "scenario": "llm",
  "n_samples": 10,
  "n_correct": 7,
  "pass_at_k": {"1": 0.7, "5": 0.998, "10": 1.0},
  "patches": [{"passed": true, "error": "", "output_snippet": "..."}]
}
```
