"""
LLM Gateway — Multi-Provider Access with Key Rotation
=======================================================

Single entry point for every LLM call in the system: MegaLLM first, then
Gemini with automatic key rotation on quota exhaustion.

FOUR BUGS THIS FIXES, all of which were silent:

1. DROPPED GENERATION PARAMETERS.
   `invoke_with_rotation` built the MegaLLM client with no arguments, so
   whenever MegaLLM was configured (the default), every caller's temperature
   and model choice was discarded. The fact-checker asked for temperature 0.0
   and got the provider default; the outline agent asked for 0.3 and got the
   same thing. Per-agent tuning only ever took effect on the Gemini fallback
   path. `generate()` now threads parameters through whichever provider wins.

2. UNSAFE ROTATION UNDER CONCURRENCY.
   `_gemini_index` was mutated without a lock. Two requests rotating at once
   could skip a key or land on the same exhausted one. Now guarded.

3. BLOCKING SLEEP ON THE EVENT LOOP.
   Retry backoff used `time.sleep`, which stalls every other request in the
   process when called from an async handler. The async path uses
   `asyncio.sleep`.

4. NO CONCURRENCY LIMIT.
   Gemini's free tier allows 15 requests/minute. Once the article pipeline
   started fanning out section writes in parallel, nothing stopped it from
   issuing them all at once and burning the whole quota in one burst. A
   semaphore now bounds in-flight calls.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# A message is either a LangChain BaseMessage or a (role, content) tuple.
MessageLike = Union[Tuple[str, str], Any]


class LLMProviderError(Exception):
    """Raised when all LLM providers fail."""
    pass


def response_text(response: Any) -> str:
    """
    Normalize an LLM response to plain text.

    Gemini 3.x returns `.content` as a list of typed blocks
    ([{"type": "text", "text": ...}, ...]) rather than a bare string.
    Handles every response shape the providers emit.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return "" if content is None else str(content)


class LLMProvider:
    """
    LLM access with provider failover and Gemini key rotation.

        provider = LLMProvider()

        # Preferred: parameters actually reach the provider
        text = provider.generate([("system", "..."), ("user", "...")], temperature=0.0)
        text = await provider.agenerate(messages, temperature=0.2)
    """

    # Errors meaning "this key is out of quota" — rotate and retry.
    EXHAUSTION_SIGNALS = [
        "resourceexhausted", "429", "quota", "rate limit",
        "too many requests", "resource has been exhausted",
    ]

    # Errors meaning "this key will never work" — rotate without backoff.
    KEY_UNUSABLE_SIGNALS = [
        "permission_denied", "403", "not_found", "404",
        "api key not valid", "invalid api key", "unauthenticated", "401",
    ]

    def __init__(self):
        settings = get_settings()
        self._mega_key: Optional[str] = settings.mega_api_key
        self._gemini_keys: List[str] = settings.gemini_keys
        self._gemini_index: int = 0
        self._settings = settings

        # Guards _gemini_index. Never held across an await or a network call.
        self._rotation_lock = threading.Lock()

        # Bounds in-flight LLM calls so parallel agent nodes can't exhaust the
        # free-tier RPM allowance in a single burst. Created lazily because an
        # asyncio.Semaphore binds to the running loop.
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

        if not self._mega_key and not self._gemini_keys:
            raise LLMProviderError(
                "No LLM API keys configured. Set MEGA_API_KEY or GOOGLE_API_KEY_* "
                "in your .env file."
            )

        providers = []
        if self._mega_key:
            providers.append(f"MegaLLM ({settings.mega_model})")
        if self._gemini_keys:
            providers.append(f"Gemini ({len(self._gemini_keys)} key(s), {settings.gemini_model})")

        logger.info(
            f"LLM providers initialized: {', '.join(providers)}",
            extra={"component": "llm"},
        )

    # ── Key management ──────────────────────────────────────────

    @property
    def current_gemini_key(self) -> str:
        if not self._gemini_keys:
            raise LLMProviderError("No Gemini keys available")
        with self._rotation_lock:
            return self._gemini_keys[self._gemini_index]

    def rotate_gemini_key(self) -> Optional[str]:
        """Advance to the next Gemini key. None if there's nowhere to go."""
        with self._rotation_lock:
            if len(self._gemini_keys) <= 1:
                return None
            self._gemini_index = (self._gemini_index + 1) % len(self._gemini_keys)
            new_key = self._gemini_keys[self._gemini_index]
            index = self._gemini_index + 1
            total = len(self._gemini_keys)

        logger.warning(
            f"Rotated to Gemini key {index}/{total}",
            extra={"component": "llm"},
        )
        return new_key

    def _is_exhaustion_error(self, error: Exception) -> bool:
        s = str(error).lower()
        return any(sig in s for sig in self.EXHAUSTION_SIGNALS)

    def _is_key_unusable_error(self, error: Exception) -> bool:
        s = str(error).lower()
        return any(sig in s for sig in self.KEY_UNUSABLE_SIGNALS)

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore capping concurrent LLM calls."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._settings.llm_max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    # ── Client construction ─────────────────────────────────────

    def create_mega_llm(self, **kwargs):
        """OpenAI-compatible client pointed at MegaLLM."""
        from langchain_openai import ChatOpenAI
        kwargs.setdefault("model", self._settings.mega_model)
        return ChatOpenAI(
            api_key=self._mega_key,
            base_url=self._settings.mega_base_url,
            **kwargs,
        )

    def create_gemini_llm(self, api_key: Optional[str] = None, **kwargs):
        """Gemini client using the given key, or the current one."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs.setdefault("model", self._settings.gemini_model)
        return ChatGoogleGenerativeAI(
            google_api_key=api_key or self.current_gemini_key,
            **kwargs,
        )

    def create_llm(self, **kwargs):
        """Best available client (MegaLLM preferred)."""
        if self._mega_key:
            return self.create_mega_llm(**kwargs)
        return self.create_gemini_llm(**kwargs)

    @staticmethod
    def _llm_kwargs(temperature: Optional[float], model: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if model is not None:
            kwargs["model"] = model
        return kwargs

    # ── Primary API ─────────────────────────────────────────────

    def generate(
        self,
        messages: Sequence[MessageLike],
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        Run a chat completion and return plain text.

        Unlike the old `invoke_with_rotation`, temperature and model reach the
        provider that actually serves the request — including MegaLLM.
        """
        kwargs = self._llm_kwargs(temperature, model)
        response = self.invoke_with_rotation(
            create_llm_fn=lambda key: self.create_gemini_llm(api_key=key, **kwargs),
            invoke_fn=lambda llm: llm.invoke(list(messages)),
            llm_kwargs=kwargs,
            max_retries=max_retries,
        )
        return response_text(response)

    async def agenerate(
        self,
        messages: Sequence[MessageLike],
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        Async chat completion.

        Concurrency-limited and non-blocking: retry backoff uses asyncio.sleep
        so waiting on one exhausted key doesn't stall the whole event loop.
        """
        async with self._get_semaphore():
            return await self._agenerate_unlimited(
                messages, temperature=temperature, model=model, max_retries=max_retries
            )

    async def _agenerate_unlimited(
        self,
        messages: Sequence[MessageLike],
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        settings = self._settings
        kwargs = self._llm_kwargs(temperature, model)
        payload = list(messages)
        retry_delay = settings.llm_retry_delay

        if max_retries is None:
            max_retries = max(len(self._gemini_keys), 1)

        # ── MegaLLM first ──
        if self._mega_key:
            try:
                llm = self.create_mega_llm(**kwargs)
                result = await llm.ainvoke(payload)
                logger.debug("MegaLLM call succeeded", extra={"component": "llm"})
                return response_text(result)
            except Exception as e:
                logger.warning(
                    f"MegaLLM failed: {e}. Falling back to Gemini.",
                    extra={"component": "llm"},
                )

        if not self._gemini_keys:
            raise LLMProviderError("MegaLLM failed and no Gemini keys configured.")

        # ── Gemini with rotation ──
        last_error: Optional[Exception] = None
        keys_tried: set = set()

        for _ in range(max_retries):
            key = self.current_gemini_key
            if key in keys_tried:
                break
            keys_tried.add(key)

            try:
                llm = self.create_gemini_llm(api_key=key, **kwargs)
                result = await llm.ainvoke(payload)
                return response_text(result)
            except Exception as e:
                exhausted = self._is_exhaustion_error(e)
                if not (exhausted or self._is_key_unusable_error(e)):
                    raise
                logger.warning(
                    f"Gemini key {'exhausted' if exhausted else 'unusable'}: {e}",
                    extra={"component": "llm"},
                )
                last_error = e
                if self.rotate_gemini_key() is None:
                    break
                if exhausted and retry_delay > 0:
                    await asyncio.sleep(retry_delay)

        raise LLMProviderError(
            f"All LLM providers failed. Tried MegaLLM + {len(keys_tried)} Gemini key(s). "
            f"Last error: {last_error}"
        ) from last_error

    # ── Lower-level API (kept for callers that build their own chains) ──

    def invoke_with_rotation(
        self,
        create_llm_fn: Callable[[str], Any],
        invoke_fn: Callable[[Any], Any],
        llm_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Any:
        """
        Invoke with failover and rotation, given caller-supplied factories.

        `llm_kwargs` is what fixes the parameter-drop bug: the MegaLLM branch
        doesn't call `create_llm_fn` (that factory takes a Gemini API key), so
        without these it built a default client and silently ignored whatever
        the caller configured. Callers that don't need custom chain wiring
        should use `generate()` instead.
        """
        settings = self._settings
        llm_kwargs = llm_kwargs or {}
        if max_retries is None:
            max_retries = max(len(self._gemini_keys), 1)
        if retry_delay is None:
            retry_delay = settings.llm_retry_delay

        # ── MegaLLM first ──
        if self._mega_key:
            try:
                mega_llm = self.create_mega_llm(**llm_kwargs)
                result = invoke_fn(mega_llm)
                logger.debug("MegaLLM call succeeded", extra={"component": "llm"})
                return result
            except Exception as e:
                logger.warning(
                    f"MegaLLM failed: {e}. Falling back to Gemini.",
                    extra={"component": "llm"},
                )

        if not self._gemini_keys:
            raise LLMProviderError("MegaLLM failed and no Gemini keys configured.")

        # ── Gemini with rotation ──
        last_error: Optional[Exception] = None
        keys_tried: set = set()

        for _ in range(max_retries):
            key = self.current_gemini_key
            if key in keys_tried:
                break
            keys_tried.add(key)

            try:
                return invoke_fn(create_llm_fn(key))
            except Exception as e:
                exhausted = self._is_exhaustion_error(e)
                if not (exhausted or self._is_key_unusable_error(e)):
                    raise
                logger.warning(
                    f"Gemini key {'exhausted' if exhausted else 'unusable'}: {e}",
                    extra={"component": "llm"},
                )
                last_error = e
                if self.rotate_gemini_key() is None:
                    break
                if exhausted and retry_delay > 0:
                    time.sleep(retry_delay)

        raise LLMProviderError(
            f"All LLM providers failed. Tried MegaLLM + {len(keys_tried)} Gemini key(s). "
            f"Last error: {last_error}"
        ) from last_error

    def status(self) -> dict:
        """Provider status for health checks and UI display."""
        return {
            "mega_available": self._mega_key is not None,
            "mega_model": self._settings.mega_model if self._mega_key else None,
            "gemini_keys_total": len(self._gemini_keys),
            "gemini_active_key_index": self._gemini_index + 1 if self._gemini_keys else 0,
            "gemini_model": self._settings.gemini_model,
            "max_concurrency": self._settings.llm_max_concurrency,
            "total_providers": (1 if self._mega_key else 0) + len(self._gemini_keys),
        }
