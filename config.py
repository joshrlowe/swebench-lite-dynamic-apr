"""Central configuration for the LLM-Adversarial APR framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    litellm_model: str
    env_key: str
    provider: str
    supports_temperature: bool = True
    litellm_model_fallback: str = ""


MODELS: dict[str, ModelSpec] = {
    "gpt53_codex": ModelSpec(
        key="gpt53_codex",
        display_name="GPT-5.3-Codex",
        litellm_model="gpt-5.3-codex",
        env_key="OPENAI_API_KEY",
        provider="openai",
        supports_temperature=False,
    ),
    "claude_sonnet": ModelSpec(
        key="claude_sonnet",
        display_name="Claude Sonnet 4.6",
        litellm_model="anthropic/claude-sonnet-4-6",
        env_key="ANTHROPIC_API_KEY",
        provider="anthropic",
    ),
    "claude_opus": ModelSpec(
        key="claude_opus",
        display_name="Claude Opus 4.6",
        litellm_model="anthropic/claude-opus-4-6",
        env_key="ANTHROPIC_API_KEY",
        provider="anthropic",
    ),
    "deepseek_r1": ModelSpec(
        key="deepseek_r1",
        display_name="DeepSeek-R1",
        litellm_model="openrouter/deepseek/deepseek-r1",
        env_key="OPENROUTER_API_KEY",
        provider="openrouter",
    ),
    "deepseek_v32": ModelSpec(
        key="deepseek_v32",
        display_name="DeepSeek-V3.2",
        litellm_model="openrouter/deepseek/deepseek-v3.2",
        env_key="OPENROUTER_API_KEY",
        provider="openrouter",
    ),
    "grok_41": ModelSpec(
        key="grok_41",
        display_name="Grok 4.1",
        litellm_model="xai/grok-4-1-fast-reasoning",
        env_key="XAI_API_KEY",
        provider="xai",
    ),
    "gemini_31_pro": ModelSpec(
        key="gemini_31_pro",
        display_name="Gemini 1.5 Pro",
        litellm_model="vertex_ai/gemini-1.5-pro",
        litellm_model_fallback="gemini/gemini-1.5-pro",
        env_key="GEMINI_API_KEY",
        provider="vertex_ai",
    ),
}

# Vertex AI requires project + location (used when provider == "vertex_ai")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "").strip()
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1").strip()

MODEL_ORDER: list[str] = [
    "gpt53_codex", "claude_sonnet", "claude_opus",
    "deepseek_r1", "deepseek_v32", "grok_41", "gemini_31_pro",
]

# ---------------------------------------------------------------------------
# Taxonomy categories (adapted from Tricky^2)
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, str] = {
    "variable_misuse": "Variable/Data Misuse",
    "logic_error": "Logic/Condition Error",
    "loop_flaw": "Loop/Iteration Flaw",
    "param_error": "Function Parameter Error",
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

NUM_SAMPLES = 10
PASS_K_VALUES = [1, 5, 10]
GENERATION_TEMPERATURE = 0.0
REPAIR_TEMPERATURE = 0.8
MAX_TOKENS = 4096
TEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Defects4J bug selection (project_id, bug_ids)
# ---------------------------------------------------------------------------

DEFECTS4J_BUGS: dict[str, list[int]] = {
    "Lang":  list(range(1, 21)),   # 20 bugs
    "Math":  list(range(1, 21)),   # 20 bugs
    "Chart": list(range(1, 11)),   # 10 bugs
    "Time":  list(range(1, 11)),   # 10 bugs
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(BASE_DIR, "benchmarks")
QUIXBUGS_DIR = os.path.join(BENCHMARKS_DIR, "quixbugs")
DEFECTS4J_DIR = os.path.join(BENCHMARKS_DIR, "defects4j")
DEFECTS4J_WORKDIR = os.path.join(BENCHMARKS_DIR, "d4j_checkouts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BUGS_DIR = os.path.join(RESULTS_DIR, "bugs")
COMPOUND_BUGS_DIR = os.path.join(RESULTS_DIR, "bugs_compound")
PATCHES_DIR = os.path.join(RESULTS_DIR, "patches")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluations")
COMPOUND_EVAL_DIR = os.path.join(RESULTS_DIR, "evaluations_compound")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
COST_LOG = os.path.join(RESULTS_DIR, "api_costs.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key(model_key: str) -> str | None:
    spec = MODELS[model_key]
    key = os.environ.get(spec.env_key, "").strip()
    return key if key else None


def get_available_models() -> list[str]:
    return [k for k in MODEL_ORDER if get_api_key(k)]


def get_vertex_credentials() -> str | None:
    """Return Vertex AI credentials: path to JSON file or JSON string."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    if key.startswith("{") and "client_email" in key:
        return key
    if os.path.isfile(key):
        with open(key) as f:
            return f.read()
    return key if key else None
