"""
Tests for the LLM gateway.

The headline case is the dropped-parameter bug: `invoke_with_rotation` built
the MegaLLM client with no arguments, so whenever MegaLLM was configured —
the default — every caller's temperature and model were silently discarded.
The agents' carefully chosen per-step temperatures only ever applied on the
Gemini fallback path.
"""

import asyncio

import pytest

from backend.app.core.config import Settings
from backend.app.services.llm import LLMProvider, LLMProviderError, response_text


class RecordingLLM:
    """Captures construction kwargs and can be told to fail."""

    def __init__(self, error=None, **kwargs):
        self.kwargs = kwargs
        self.error = error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return type("R", (), {"content": "ok"})()

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return type("R", (), {"content": "ok"})()


@pytest.fixture
def provider(monkeypatch):
    """
    Provider with both MegaLLM and three Gemini keys, and both client
    constructors replaced by recorders.
    """
    settings = Settings(
        mega_api_key="mega-key",
        google_api_key_1="gem-1",
        google_api_key_2="gem-2",
        google_api_key_3="gem-3",
    )
    monkeypatch.setattr("backend.app.services.llm.get_settings", lambda: settings)

    p = LLMProvider()
    p.built = []          # every client constructed, in order

    def make(kind, error_for=None):
        def factory(*args, **kwargs):
            api_key = kwargs.pop("google_api_key", None) or (args[0] if args else None)
            err = error_for(api_key) if error_for else None
            llm = RecordingLLM(error=err, **kwargs)
            p.built.append((kind, api_key, kwargs))
            return llm
        return factory

    p._make = make
    p.create_mega_llm = make("mega")
    p.create_gemini_llm = lambda api_key=None, **kw: make("gemini")(api_key, **kw)
    return p


class TestResponseNormalization:
    def test_plain_string(self):
        assert response_text(type("R", (), {"content": "hello"})()) == "hello"

    def test_gemini_typed_blocks(self):
        """Gemini 3.x returns a list of typed blocks, not a bare string."""
        r = type("R", (), {"content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]})()
        assert response_text(r) == "Hello world"

    def test_mixed_list(self):
        r = type("R", (), {"content": ["a", {"text": "b"}]})()
        assert response_text(r) == "ab"

    def test_none(self):
        assert response_text(type("R", (), {"content": None})()) == ""

    def test_bare_string_input(self):
        assert response_text("raw") == "raw"


class TestParameterPassThrough:
    """The regression that motivated this rewrite."""

    def test_generate_passes_temperature_to_megallm(self, provider):
        provider.generate([("user", "hi")], temperature=0.0)
        kind, _, kwargs = provider.built[0]
        assert kind == "mega"
        assert kwargs.get("temperature") == 0.0, \
            "temperature must reach MegaLLM, not be dropped"

    def test_generate_passes_model_override(self, provider):
        provider.generate([("user", "hi")], temperature=0.3, model="custom-model")
        _, _, kwargs = provider.built[0]
        assert kwargs.get("model") == "custom-model"

    def test_invoke_with_rotation_passes_llm_kwargs(self, provider):
        provider.invoke_with_rotation(
            create_llm_fn=lambda key: provider.create_gemini_llm(api_key=key, temperature=0.9),
            invoke_fn=lambda llm: llm.invoke([]),
            llm_kwargs={"temperature": 0.9},
        )
        kind, _, kwargs = provider.built[0]
        assert kind == "mega"
        assert kwargs.get("temperature") == 0.9

    async def test_agenerate_passes_temperature(self, provider):
        await provider.agenerate([("user", "hi")], temperature=0.7)
        _, _, kwargs = provider.built[0]
        assert kwargs.get("temperature") == 0.7

    def test_distinct_temperatures_are_not_conflated(self, provider):
        provider.generate([("user", "a")], temperature=0.0)
        provider.generate([("user", "b")], temperature=0.3)
        assert [k.get("temperature") for _, _, k in provider.built] == [0.0, 0.3]


class TestFailoverAndRotation:
    def test_megallm_tried_first(self, provider):
        provider.generate([("user", "hi")], temperature=0.1)
        assert provider.built[0][0] == "mega"

    def test_falls_back_to_gemini_when_mega_fails(self, provider):
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("mega down"))
        assert provider.generate([("user", "hi")], temperature=0.1) == "ok"
        assert [b[0] for b in provider.built] == ["mega", "gemini"]

    def test_rotates_on_quota_exhaustion(self, provider):
        """429 on key 1 should advance to key 2, not give up."""
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("mega down"))
        provider.create_gemini_llm = lambda api_key=None, **kw: provider._make(
            "gemini",
            lambda k: RuntimeError("429 quota exceeded") if k == "gem-1" else None,
        )(api_key, **kw)

        assert provider.generate([("user", "hi")], temperature=0.1) == "ok"
        gemini_keys = [b[1] for b in provider.built if b[0] == "gemini"]
        assert gemini_keys == ["gem-1", "gem-2"]

    def test_rotates_past_unusable_key(self, provider):
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("down"))
        provider.create_gemini_llm = lambda api_key=None, **kw: provider._make(
            "gemini",
            lambda k: RuntimeError("401 invalid api key") if k == "gem-1" else None,
        )(api_key, **kw)

        assert provider.generate([("user", "hi")], temperature=0.1) == "ok"

    def test_non_quota_error_propagates(self, provider):
        """A genuine bug shouldn't be masked by burning through every key."""
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("down"))
        provider.create_gemini_llm = lambda api_key=None, **kw: provider._make(
            "gemini", lambda k: ValueError("malformed request")
        )(api_key, **kw)

        with pytest.raises(ValueError, match="malformed request"):
            provider.generate([("user", "hi")], temperature=0.1)

    def test_raises_when_all_keys_exhausted(self, provider):
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("down"))
        provider.create_gemini_llm = lambda api_key=None, **kw: provider._make(
            "gemini", lambda k: RuntimeError("429 quota exceeded")
        )(api_key, **kw)
        provider._settings = provider._settings.model_copy(update={"llm_retry_delay": 0})

        with pytest.raises(LLMProviderError, match="All LLM providers failed"):
            provider.generate([("user", "hi")], temperature=0.1)

    def test_rotation_is_bounded_by_key_count(self, provider):
        provider.create_mega_llm = provider._make("mega", lambda k: RuntimeError("down"))
        provider.create_gemini_llm = lambda api_key=None, **kw: provider._make(
            "gemini", lambda k: RuntimeError("429 quota")
        )(api_key, **kw)
        provider._settings = provider._settings.model_copy(update={"llm_retry_delay": 0})

        with pytest.raises(LLMProviderError):
            provider.generate([("user", "hi")], temperature=0.1)

        tried = [b[1] for b in provider.built if b[0] == "gemini"]
        assert len(tried) == len(set(tried)) == 3


class TestConcurrencyControl:
    async def test_semaphore_bounds_parallel_calls(self, provider):
        """
        Free-tier Gemini allows 15 RPM. Parallel section writes must not be
        able to fire all at once.
        """
        provider._settings = provider._settings.model_copy(
            update={"llm_max_concurrency": 2}
        )
        provider._semaphore = None

        in_flight = peak = 0

        class SlowLLM(RecordingLLM):
            async def ainvoke(self, messages):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.02)
                in_flight -= 1
                return type("R", (), {"content": "ok"})()

        provider.create_mega_llm = lambda **kw: SlowLLM(**kw)

        await asyncio.gather(*[
            provider.agenerate([("user", f"q{i}")], temperature=0.1) for i in range(8)
        ])
        assert peak <= 2

    def test_rotation_is_thread_safe(self, provider):
        """_gemini_index was previously mutated without a lock."""
        import threading

        def rotate_many():
            for _ in range(200):
                provider.rotate_gemini_key()

        threads = [threading.Thread(target=rotate_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert 0 <= provider._gemini_index < len(provider._gemini_keys)


class TestConfiguration:
    def test_requires_at_least_one_key(self, monkeypatch):
        monkeypatch.setattr(
            "backend.app.services.llm.get_settings",
            lambda: Settings(mega_api_key=None, google_api_key=None,
                             google_api_key_1=None, google_api_key_2=None,
                             google_api_key_3=None),
        )
        with pytest.raises(LLMProviderError, match="No LLM API keys"):
            LLMProvider()

    def test_status_reports_providers(self, provider):
        status = provider.status()
        assert status["mega_available"] is True
        assert status["gemini_keys_total"] == 3
        assert status["total_providers"] == 4
        assert "max_concurrency" in status
