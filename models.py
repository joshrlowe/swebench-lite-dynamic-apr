from __future__ import annotations

import json, logging, os, time, anthropic, openai, yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NoReturn

# Our settings for a single LLM model
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    provider: str
    api_key_env: str
    temperature: float
    top_p: float
    max_tokens: int

# Timeouts and number of times we can retry for running code in Docker
@dataclass(frozen=True)
class ExecutionConfig:
    python_timeout_sec: int
    max_injection_retries: int

# A place to store different parameters we can adjust for our experiment
@dataclass(frozen=True)
class ExperimentConfig:
    pass_k_values: tuple[int, ...]
    n_samples: int
    target_cases_per_model_per_bug_type: int

# The configuration for the models that we use during experimentation, the different execution parameters, and the experiment configuration
@dataclass(frozen=True)
class AppConfig:
    models: dict[str, ModelConfig]
    execution: ExecutionConfig
    experiment: ExperimentConfig

# Parse the configuration file and build an AppConfig (cached)
@lru_cache(maxsize=8)
def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    p = Path(path)
    cfg_path = p if p.is_absolute() or p.exists() else Path(__file__).resolve().parents[2] / p
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

        models = { name: ModelConfig( model_id=m["model_id"], provider=m["provider"], api_key_env=m["api_key_env"], temperature=float(m["temperature"]), top_p=float(m["top_p"]), max_tokens=int(m["max_tokens"]), ) for name, m in raw["models"].items() }

    ex = raw["execution"]
    execution = ExecutionConfig( python_timeout_sec=int(ex["python_timeout_sec"]), max_injection_retries=int(ex["max_injection_retries"]), )

    exp = raw["experiment"]
    experiment = ExperimentConfig( pass_k_values=tuple(int(k) for k in exp["pass_k_values"]), n_samples=int(exp["n_samples"]), target_cases_per_model_per_bug_type=int(exp["target_cases_per_model_per_bug_type"]), )

    return AppConfig(models=models, execution=execution, experiment=experiment)

# Load .env vars into os.environ without overwriting existing ones
def load_project_dotenv(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val

ALL_MODEL_NAMES = [ "grok-4.20", "gpt-5.4-codex", "claude-sonnet-4.6", "claude-opus-4.6", "gemini-3.1-pro", ]

N_REPAIR_SAMPLES_DEFAULT = load_config().experiment.n_samples

# Get the target file path out of a unified diff following Hugging Face recommendations
def _extract_file_path_from_patch(patch_text):
    from unidiff import PatchSet
    patchset = PatchSet(patch_text)
    if len(patchset) != 1:
        return None
    target = patchset[0].target_file
    if target.startswith("b/"):
        target = target[2:]
    return target

# Get file content from GitHub at a specific commit
def _get_file_content_at_commit(repo_url, commit, file_path):
    import urllib.request
    owner_repo = repo_url.rstrip("/")
    if owner_repo.startswith("https://github.com/"):
        owner_repo = owner_repo[len("https://github.com/"):]
    raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{file_path}"
    with urllib.request.urlopen(raw_url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")

# Apply a unified diff to file content in memory
def _apply_patch_to_content(original, patch_text, file_path):
    from unidiff import PatchSet
    patchset = PatchSet(patch_text)
    lines = original.splitlines(keepends=True)
    if not lines or not lines[-1].endswith("\n"):
        lines = [l + "\n" for l in original.splitlines()]
    for patched_file in patchset:
        target = patched_file.target_file
        if target.startswith("b/"):
            target = target[2:]
        if target != file_path:
            continue
        result_lines = list(lines)
        offset = 0
        for hunk in patched_file:
            start = hunk.source_start - 1 + offset
            src_len = hunk.source_length
            new_lines = []
            for line in hunk:
                if line.is_added or line.is_context:
                    val = line.value
                    if not val.endswith("\n"):
                        val += "\n"
                    new_lines.append(val)
            result_lines[start:start + src_len] = new_lines
            offset += len(new_lines) - src_len
        return "".join(result_lines)
    return None

# Load and hydrate the selected SWE-bench instances with file content
def load_swebench_cases(instances_file="benchmarks/swebench_selected_instances.json"):
    from datasets import load_dataset
    from src.prompts.inject import BUG_TYPES

    instances_path = Path(__file__).resolve().parents[2] / instances_file
    selected_ids = json.loads(instances_path.read_text())
    print(f"Loading {len(selected_ids)} SWE-bench Lite instances...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    ds_map = {rec["instance_id"]: rec for rec in ds}

    cases = []
    for i, iid in enumerate(sorted(selected_ids)):
        rec = ds_map.get(iid)
        if rec is None:
            print(f"  {iid}: not found in dataset, skipping")
            continue
        file_path = _extract_file_path_from_patch(rec["patch"])
        if file_path is None:
            print(f"  {iid}: cannot parse patch, skipping")
            continue
        repo_url = f"https://github.com/{rec['repo']}"
        buggy_content = _get_file_content_at_commit(repo_url, rec["base_commit"], file_path)
        if buggy_content is None:
            continue
        correct_content = _apply_patch_to_content(buggy_content, rec["patch"], file_path)
        if correct_content is None:
            continue
        bug_type = BUG_TYPES[len(cases) % len(BUG_TYPES)]
        cases.append({ "id": len(cases) + 1, "instance_id": iid, "repo": rec["repo"], "problem_statement": rec["problem_statement"], "base_commit": rec["base_commit"], "gold_patch": rec["patch"], "buggy_file_path": file_path, "buggy_file_content": buggy_content, "correct_file_content": correct_content, "fail_to_pass": rec["FAIL_TO_PASS"], "pass_to_pass": rec["PASS_TO_PASS"], "version": rec.get("version"), "environment_setup_commit": rec.get("environment_setup_commit"), "assigned_bug_type": bug_type, })
        print(f"  [{i+1}/{len(selected_ids)}] {iid}: loaded ({bug_type})", flush=True)
    print(f"Loaded {len(cases)}/{len(selected_ids)} instances")
    return cases

max_retries = 3
request_timeout_time = 120.0
backoff_power = 2.0

# Encapsualtes a LLM completion and token usage statistcs
@dataclass
class ModelResponse:
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    raw_response: dict

# Base exception for anything that goes wrong talking to a model
class ModelClientError(Exception):
    pass

# Error where we havea API key or missing permissions
class AuthError(ModelClientError):
    pass

# Erorr where we couldn't make sense of the model's response format
class ParseError(ModelClientError):
    pass

# Rate limiting error
class RateLimitError(ModelClientError):
    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after

# Generic model error
class ModelError(ModelClientError):

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable

# Try to pull a retry-after header value from an SDK exception
def _parse_retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

# Convert openai/anthropic SDK exceptions into our error hierarchy
def map_sdk_exception(exc: BaseException) -> ModelClientError:
    if isinstance(exc, ModelClientError):
        return exc

    if isinstance(exc, openai.AuthenticationError):
        return AuthError(str(exc))
    if isinstance(exc, openai.PermissionDeniedError):
        return AuthError(str(exc))
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(str(exc), retry_after=_parse_retry_after(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return ModelError(str(exc), retryable=True)
    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "status_code", None)
        if code in (401, 403):
            return AuthError(str(exc))
        if code == 429:
            return RateLimitError(str(exc), retry_after=_parse_retry_after(exc))
        if code is not None and code >= 500:
            return ModelError(str(exc), retryable=True)
        return ModelError(str(exc), retryable=False)
    if isinstance(exc, openai.APIError):
        return ModelError(str(exc), retryable=False)

    if isinstance(exc, anthropic.AuthenticationError):
        return AuthError(str(exc))
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AuthError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc), retry_after=_parse_retry_after(exc))
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return ModelError(str(exc), retryable=True)
    if isinstance(exc, anthropic.APIStatusError):
        code = getattr(exc, "status_code", None)
        if code in (401, 403):
            return AuthError(str(exc))
        if code == 429:
            return RateLimitError(str(exc), retry_after=_parse_retry_after(exc))
        if code is not None and code >= 500:
            return ModelError(str(exc), retryable=True)
        return ModelError(str(exc), retryable=False)
    if isinstance(exc, anthropic.APIError):
        return ModelError(str(exc), retryable=False)

    err_s = str(exc).lower()
    if "401" in err_s or "403" in err_s or "unauthenticated" in err_s:
        return AuthError(str(exc))
    if "429" in err_s or "rate limit" in err_s or "resource exhausted" in err_s:
        return RateLimitError(str(exc))

    return ModelError(str(exc), retryable=False)

# Rraise any exception as a ModelClientError
def rethrow_as_model_error(exc: BaseException) -> NoReturn:
    if isinstance(exc, ModelClientError):
        raise exc
    mapped = map_sdk_exception(exc)
    raise mapped from exc

# Extract the text content from an OpenAI chat completion choice
def openai_chat_message_text(choice: Any) -> str:
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p.get("text") or ""))
            elif hasattr(p, "text"):
                t = getattr(p, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""

# Pull the text out of an Anthropic messages response
def anthropic_response_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    if not blocks:
        raise ParseError("Anthropic response has no content blocks")
    parts: list[str] = []
    for block in blocks:
        if hasattr(block, "text"):
            t = getattr(block, "text", None)
            if isinstance(t, str) and t:
                parts.append(t)
    if not parts and blocks:
        raise ParseError("Anthropic response has no text blocks")
    return "".join(parts)

# Serialize an Anthropic response into a plain dict for logging
def build_anthropic_raw_dict(response: Any) -> dict:
    return { "id": response.id, "model": response.model, "content": [ {"text": getattr(c, "text", ""), "type": getattr(c, "type", "")} for c in (getattr(response, "content", None) or []) ], "usage": { "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens, }, "stop_reason": response.stop_reason, }

# Call the OpenAI API n times sequentially when n>1 isn't supported
def sequential_openai_chat_completions( client: Any, *, model_id: str, logical_model_name: str, prompt: str, n: int, temperature: float, top_p: float, max_tokens: int, log: logging.Logger, use_max_completion_tokens: bool = False, extra_create_fields: dict[str, Any] | None = None, ) -> list[ModelResponse]:
    token_kw: dict[str, int] = ( {"max_completion_tokens": max_tokens} if use_max_completion_tokens else {"max_tokens": max_tokens} )
    results: list[ModelResponse] = []
    for i in range(n):
        try:
            create_kwargs: dict[str, Any] = dict( model=model_id, messages=[{"role": "user", "content": prompt}], n=1, temperature=temperature, top_p=top_p, **token_kw, )
            if extra_create_fields:
                create_kwargs.update(extra_create_fields)
            response = client.chat.completions.create(**create_kwargs)
        except Exception as e:
            rethrow_as_model_error(e)
        choice = response.choices[0]
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        log.debug( "%s | sequential call %d/%d | prompt_tokens=%d | completion_tokens=%d", logical_model_name, i + 1, n, prompt_tokens, completion_tokens, )
        results.append( ModelResponse( text=openai_chat_message_text(choice), model_name=logical_model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, raw_response=response.model_dump(), ) )
    return results

# Abstract base for all LLM clients (Anthropic, OpenAI, etc.)
class ModelClient(ABC):
    def __init__( self, model_name: str, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 2048, *, request_timeout: float | None = None, ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout = ( float(request_timeout) if request_timeout is not None else request_timeout_time )

    @abstractmethod
    def complete(self, prompt: str, n: int = 1) -> list[ModelResponse]:
        pass

    # Call complete() with exponential backoff on transient failures
    def complete_with_retry(self, prompt: str, n: int = 1, max_retries: int | None = None) -> list[ModelResponse]:
        attempts = max_retries if max_retries is None else max_retries
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return self.complete(prompt, n=n)
            except ModelClientError as e:
                last_exc = e
                if isinstance(e, (AuthError, ParseError)):
                    raise
                if isinstance(e, RateLimitError) or (isinstance(e, ModelError) and e.retryable):
                    retry_after = getattr(e, "retry_after", None)
                    wait = float(retry_after) if retry_after is not None else backoff_power ** attempt
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                last_exc = e
                error_str = str(e).lower()
                if any( code in error_str for code in ("401", "403", "400", "invalid_request") ):
                    raise
                wait = backoff_power**attempt
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

# Zero out prompt_tokens on all but the first response to avoid double-counting
def dedupe_prompt_tokens(responses: list[ModelResponse]) -> list[ModelResponse]:
    if len(responses) <= 1:
        return responses
    first = responses[0]
    rest = [ ModelResponse( text=r.text, model_name=r.model_name, prompt_tokens=0, completion_tokens=r.completion_tokens, raw_response=r.raw_response, ) for r in responses[1:] ]
    return [first, *rest]

# Client for Anthropic's messages API
class AnthropicClient(ModelClient):
    def __init__( self, model_name: str, model_id: str, api_key: str, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 2048, ):
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_id = model_id
        self._client = anthropic.Anthropic( api_key=api_key, timeout=self.request_timeout, )

    # Send prompt to Claude and collect n responses sequentially
    def complete(self, prompt: str, n: int = 1) -> list[ModelResponse]:

        results: list[ModelResponse] = []
        for i in range(n):
            try:
                response = self._client.messages.create( model=self.model_id, max_tokens=self.max_tokens, temperature=self.temperature, messages=[{"role": "user", "content": prompt}], )
            except Exception as e:
                rethrow_as_model_error(e)

            text = anthropic_response_text(response)
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            raw = build_anthropic_raw_dict(response)

            results.append( ModelResponse( text=text, model_name=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, raw_response=raw, ) )

        return results

# Client for OpenAI-compatible APIs (GPT, Grok, Gemini via OpenRouter, etc.)
class OpenAICompatibleClient(ModelClient):
    reasoning_cap = {"extra_body": {"reasoning": {"max_tokens": 2048}}}
    reasoning_models = {"gemini-3.1-pro", "grok-4.20"}

    def __init__( self, model_name: str, model_id: str, api_key: str, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 2048, base_url: str | None = None, ):
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_id = model_id
        self._base_url = base_url
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.request_timeout}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

    def _use_max_completion_tokens(self) -> bool:
        if self._base_url is not None:
            return False
        mid = self.model_id.lower()
        return mid.startswith("gpt-5") or mid.startswith("o1") or mid.startswith("o3")

    def _max_output_kw(self) -> dict[str, int]:
        if self._use_max_completion_tokens():
            return {"max_completion_tokens": self.max_tokens}
        return {"max_tokens": self.max_tokens}

    # Add reasoning cap for models that support it on openrouter
    def _extra_create_fields(self) -> dict | None:
        if self._base_url is not None and self.model_name in self.reasoning_models:
            return self.reasoning_cap
        return None

    # Request n completions, falling back to sequential if the API won't batch
    def complete(self, prompt: str, n: int = 1) -> list[ModelResponse]:
        create_kwargs: dict[str, Any] = dict(model=self.model_id, messages=[{"role": "user", "content": prompt}], n=n, temperature=self.temperature, top_p=self.top_p, **self._max_output_kw())
        extra = self._extra_create_fields()
        if extra:
            create_kwargs.update(extra)
        response = self._client.chat.completions.create(**create_kwargs)

        choices = response.choices

        if len(choices) < n:
            return sequential_openai_chat_completions( self._client, model_id=self.model_id, logical_model_name=self.model_name, prompt=prompt, n=n, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, use_max_completion_tokens=self._use_max_completion_tokens(), extra_create_fields=self._extra_create_fields(), )

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        raw = response.model_dump()

        return dedupe_prompt_tokens([ ModelResponse( text=openai_chat_message_text(choice), model_name=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, raw_response=raw, ) for choice in choices ])


# Build the right ModelClient subclass depending on the model name
def get_client(model_name: str, config_path: str = "config/config.yaml") -> ModelClient:
    cfg = load_config(config_path)

    if model_name not in cfg.models:
        raise ValueError( f"Unknown model: {model_name}. Available: {list(cfg.models.keys())}" )

    model_cfg = cfg.models[model_name]
    provider = model_cfg.provider

    api_key = os.environ.get(model_cfg.api_key_env)
    if not api_key:
        raise EnvironmentError( f"Missing env var: {model_cfg.api_key_env} for model {model_name}" )

        kwargs = dict( model_name=model_name, model_id=model_cfg.model_id, api_key=api_key, temperature=model_cfg.temperature, top_p=model_cfg.top_p, max_tokens=model_cfg.max_tokens, )

    if provider == "openrouter":
        return OpenAICompatibleClient(**kwargs, base_url="https://openrouter.ai/api/v1")
    elif provider == "openai":
        return OpenAICompatibleClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
