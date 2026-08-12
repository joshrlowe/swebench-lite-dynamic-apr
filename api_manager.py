"""Module 1: Unified API Manager using litellm with GPT-5.3-Codex fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import litellm
import openai

import config

logger = logging.getLogger(__name__)

litellm.drop_params = True  # silently drop unsupported params per provider


@dataclass
class UsageRecord:
    model_key: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class APIManager:
    """Unified async interface to all 7 frontier LLMs."""

    PRICING_PER_1K: dict[str, tuple[float, float]] = {
        "gpt53_codex":   (0.002, 0.008),
        "claude_sonnet":  (0.003, 0.015),
        "claude_opus":    (0.015, 0.075),
        "deepseek_r1":    (0.001, 0.004),
        "deepseek_v32":   (0.001, 0.004),
        "grok_41":        (0.003, 0.015),
        "gemini_31_pro":  (0.00125, 0.005),
    }

    def __init__(self, concurrency: int = 5) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._usage: list[UsageRecord] = []
        self._openai_client: openai.AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = config.MAX_TOKENS,
        n_samples: int = 1,
        system: str = "You are an expert software engineer.",
    ) -> list[str]:
        """Return *n_samples* response texts from *model_key*."""
        spec = config.MODELS[model_key]
        api_key = config.get_api_key(model_key)
        if not api_key:
            raise ValueError(f"No API key set for {spec.display_name} (env: {spec.env_key})")

        tasks = [
            self._call(spec, prompt, temperature, max_tokens, system, api_key)
            for _ in range(n_samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[str] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("API call failed for %s: %s", model_key, r)
                outputs.append("")
            else:
                outputs.append(r)
        return outputs

    def get_available_models(self) -> list[str]:
        return config.get_available_models()

    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self._usage)

    def flush_costs(self) -> None:
        """Append accumulated usage records to the JSONL cost log."""
        if not self._usage:
            return
        import os
        os.makedirs(os.path.dirname(config.COST_LOG), exist_ok=True)
        with open(config.COST_LOG, "a") as f:
            for rec in self._usage:
                f.write(json.dumps({
                    "model": rec.model_key,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "cost_usd": rec.cost_usd,
                    "timestamp": rec.timestamp,
                }) + "\n")
        self._usage.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call(
        self,
        spec: config.ModelSpec,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str,
        api_key: str,
    ) -> str:
        async with self._semaphore:
            for attempt in range(4):
                try:
                    if spec.key == "gpt53_codex":
                        return await self._call_openai_responses(
                            spec, prompt, max_tokens, system, api_key
                        )
                    return await self._call_litellm(
                        spec, prompt, temperature, max_tokens, system, api_key
                    )
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s – retrying in %ds",
                        attempt + 1, 4, spec.display_name, exc, wait,
                    )
                    await asyncio.sleep(wait)
            raise RuntimeError(f"All retries exhausted for {spec.display_name}")

    async def _call_litellm(
        self,
        spec: config.ModelSpec,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str,
        api_key: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict = {
            "model": spec.litellm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if spec.provider == "openrouter":
            kwargs["api_key"] = api_key
        elif spec.provider == "xai":
            kwargs["api_key"] = api_key
            kwargs["api_base"] = "https://api.x.ai/v1"
        elif spec.provider == "anthropic":
            kwargs["api_key"] = api_key
        elif spec.provider == "vertex_ai":
            creds = config.get_vertex_credentials()
            try:
                import json
                if creds and creds.strip().startswith("{") and json.loads(creds):
                    kwargs["vertex_credentials"] = creds
            except (json.JSONDecodeError, ValueError):
                pass
            if creds and not kwargs.get("vertex_credentials") and spec.litellm_model_fallback:
                kwargs["model"] = spec.litellm_model_fallback
                kwargs["api_key"] = api_key
                kwargs.pop("vertex_credentials", None)
                kwargs.pop("vertex_project", None)
                kwargs.pop("vertex_location", None)
            elif kwargs.get("vertex_credentials"):
                if config.VERTEX_PROJECT:
                    kwargs["vertex_project"] = config.VERTEX_PROJECT
                if config.VERTEX_LOCATION:
                    kwargs["vertex_location"] = config.VERTEX_LOCATION
        elif spec.provider == "google":
            kwargs["api_key"] = api_key

        logger.debug("litellm call: model=%s temp=%.1f", spec.litellm_model, temperature)
        response = await litellm.acompletion(**kwargs)
        text = response.choices[0].message.content or ""

        prompt_tok = getattr(response.usage, "prompt_tokens", 0)
        completion_tok = getattr(response.usage, "completion_tokens", 0)
        self._log_usage(spec.key, prompt_tok, completion_tok)
        return text

    async def _call_openai_responses(
        self,
        spec: config.ModelSpec,
        prompt: str,
        max_tokens: int,
        system: str,
        api_key: str,
    ) -> str:
        """GPT-5.3-Codex requires the OpenAI Responses API."""
        if self._openai_client is None:
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)

        logger.debug("OpenAI Responses API call: model=%s", spec.litellm_model)
        response = await self._openai_client.responses.create(
            model=spec.litellm_model,
            instructions=system,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        text = response.output_text or ""

        prompt_tok = getattr(response.usage, "input_tokens", 0)
        completion_tok = getattr(response.usage, "output_tokens", 0)
        self._log_usage(spec.key, prompt_tok, completion_tok)
        return text

    def _log_usage(self, model_key: str, prompt_tokens: int, completion_tokens: int) -> None:
        p_rate, c_rate = self.PRICING_PER_1K.get(model_key, (0.0, 0.0))
        cost = (prompt_tokens / 1000) * p_rate + (completion_tokens / 1000) * c_rate
        self._usage.append(UsageRecord(
            model_key=model_key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        ))
        logger.debug(
            "Usage: %s – %d prompt + %d completion tokens – $%.4f",
            model_key, prompt_tokens, completion_tokens, cost,
        )
