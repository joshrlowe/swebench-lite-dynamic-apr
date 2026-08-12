"""Tests for APIManager — model routing, retries, and cost tracking.

All API calls are mocked. No real API spending.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from api_manager import APIManager, UsageRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    return APIManager(concurrency=2)


@pytest.fixture
def mock_env(monkeypatch):
    """Set fake API keys for all models."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-openrouter")
    monkeypatch.setenv("XAI_API_KEY", "sk-fake-xai")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

class TestModelRouting:
    def test_all_models_in_config(self):
        """All 7 expected models should be in config.MODELS."""
        expected = {
            "gpt53_codex", "claude_sonnet", "claude_opus",
            "deepseek_r1", "deepseek_v32", "grok_41", "gemini_31_pro",
        }
        assert set(config.MODELS.keys()) == expected

    def test_model_order_matches_models(self):
        """MODEL_ORDER should reference only valid model keys."""
        for key in config.MODEL_ORDER:
            assert key in config.MODELS

    def test_available_models_requires_keys(self, monkeypatch):
        """Models without API keys should not be 'available'."""
        # Clear all keys
        for spec in config.MODELS.values():
            monkeypatch.delenv(spec.env_key, raising=False)
        assert config.get_available_models() == []

    def test_available_models_with_keys(self, mock_env):
        """Models with API keys should be available."""
        available = config.get_available_models()
        assert len(available) == 7


# ---------------------------------------------------------------------------
# API call routing (mocked)
# ---------------------------------------------------------------------------

class TestAPICallRouting:
    @pytest.mark.asyncio
    async def test_gpt53_uses_responses_api(self, api, mock_env):
        """GPT-5.3-Codex should route through _call_openai_responses."""
        with patch.object(api, "_call_openai_responses", new_callable=AsyncMock) as mock_resp:
            mock_resp.return_value = "fixed code"
            results = await api.generate("gpt53_codex", "fix this", n_samples=1)
            assert mock_resp.called
            assert results == ["fixed code"]

    @pytest.mark.asyncio
    async def test_claude_uses_litellm(self, api, mock_env):
        """Claude models should route through _call_litellm."""
        with patch.object(api, "_call_litellm", new_callable=AsyncMock) as mock_lit:
            mock_lit.return_value = "fixed code"
            results = await api.generate("claude_opus", "fix this", n_samples=1)
            assert mock_lit.called
            assert results == ["fixed code"]

    @pytest.mark.asyncio
    async def test_deepseek_uses_litellm(self, api, mock_env):
        """DeepSeek models should route through _call_litellm."""
        with patch.object(api, "_call_litellm", new_callable=AsyncMock) as mock_lit:
            mock_lit.return_value = "fixed"
            results = await api.generate("deepseek_v32", "fix this", n_samples=1)
            assert mock_lit.called

    @pytest.mark.asyncio
    async def test_n_samples_produces_n_results(self, api, mock_env):
        """generate() with n_samples=3 should return 3 results."""
        with patch.object(api, "_call_litellm", new_callable=AsyncMock) as mock_lit:
            mock_lit.return_value = "patch"
            results = await api.generate("claude_sonnet", "fix", n_samples=3)
            assert len(results) == 3
            assert mock_lit.call_count == 3


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_failure(self, api, mock_env):
        """Should retry up to 4 times on transient errors."""
        call_count = 0

        async def flaky_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit")
            return "success"

        with patch.object(api, "_call_litellm", side_effect=flaky_call):
            results = await api.generate("claude_sonnet", "fix", n_samples=1)
            assert results == ["success"]
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_returns_empty_on_exhausted_retries(self, api, mock_env):
        """After 4 failures, should return empty string (not crash)."""
        async def always_fail(*args, **kwargs):
            raise Exception("persistent error")

        with patch.object(api, "_call_litellm", side_effect=always_fail):
            results = await api.generate("claude_sonnet", "fix", n_samples=1)
            # generate() catches exceptions and returns ""
            assert results == [""]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class TestCostTracking:
    def test_initial_cost_is_zero(self, api):
        assert api.total_cost() == 0.0

    def test_log_usage_increments_cost(self, api):
        api._log_usage("claude_sonnet", prompt_tokens=1000, completion_tokens=500)
        cost = api.total_cost()
        # claude_sonnet: (1000/1000)*0.003 + (500/1000)*0.015 = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105, abs=1e-6)

    def test_log_multiple_models(self, api):
        api._log_usage("deepseek_r1", prompt_tokens=1000, completion_tokens=1000)
        api._log_usage("claude_opus", prompt_tokens=1000, completion_tokens=1000)
        # deepseek_r1: (1)*0.001 + (1)*0.004 = 0.005
        # claude_opus: (1)*0.015 + (1)*0.075 = 0.090
        assert api.total_cost() == pytest.approx(0.095, abs=1e-6)

    def test_flush_costs_writes_jsonl(self, api, tmp_path):
        """flush_costs should write usage records to disk."""
        cost_log = str(tmp_path / "costs.jsonl")
        with patch.object(config, "COST_LOG", cost_log):
            api._log_usage("gpt53_codex", 100, 50)
            api.flush_costs()

        import json
        with open(cost_log) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 1
        assert records[0]["model"] == "gpt53_codex"
        assert records[0]["prompt_tokens"] == 100

    def test_flush_clears_usage(self, api, tmp_path):
        """After flush, total_cost should reset to 0."""
        cost_log = str(tmp_path / "costs.jsonl")
        with patch.object(config, "COST_LOG", cost_log):
            api._log_usage("gpt53_codex", 100, 50)
            api.flush_costs()
        assert api.total_cost() == 0.0

    def test_pricing_covers_all_models(self):
        """Every model key should have pricing defined."""
        for key in config.MODELS:
            assert key in APIManager.PRICING_PER_1K, f"Missing pricing for {key}"


# ---------------------------------------------------------------------------
# Missing API key handling
# ---------------------------------------------------------------------------

class TestMissingAPIKey:
    @pytest.mark.asyncio
    async def test_raises_on_missing_key(self, api, monkeypatch):
        """generate() should raise ValueError if API key is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            await api.generate("claude_opus", "fix this", n_samples=1)
