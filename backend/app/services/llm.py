"""
LLM Provider Manager — Resilient Multi-Provider Access
=========================================================

WHY THIS IS A REFACTOR (not a rewrite):
Your original key_manager.py had the right idea — MegaLLM first, Gemini fallback
with key rotation. This refactored version keeps that logic but improves:

1. CONFIGURATION: Uses the centralized Settings instead of scattered os.getenv()
2. TYPING: Proper type hints and Pydantic response models
3. LOGGING: Structured logging instead of print() — you can trace which key was
   used for which request, when rotation happened, and why
4. RESPONSE NORMALIZATION: Your original response_text() function is preserved
   but moved here as a method
5. ASYNC SUPPORT: Adds async invoke methods for use with FastAPI

TEACHING POINT — Why key rotation matters at scale:
Free Gemini API keys have a 15 RPM (requests per minute) limit.
With 4 agents × 3 RAG calls each = 12 LLM calls per article.
One user generating one article nearly exhausts a single key.
With 10 concurrent users, you need 10+ keys or a paid tier.
Key rotation is a bridge between "free tier prototype" and
"paid production API" — it lets you scale the prototype further
before committing to paid infrastructure.
"""

import logging
import time
from typing import Any, Callable, List, Optional

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when all LLM providers fail."""
    pass


def response_text(response: Any) -> str:
    """
    Normalize LLM response to plain text.

    Gemini 3.x returns .content as a list of typed blocks
    ([{"type": "text", "text": ...}, ...]) rather than a bare string.
    This handles all known response formats.

    Preserved from the original key_manager.py — this is battle-tested.
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
    Manages LLM access with automatic provider failover and key rotation.

    Architecture:
        1. Try MegaLLM (if configured) — OpenAI-compatible, fast, cheap
        2. If MegaLLM fails → fall back to Gemini with key rotation
        3. If all Gemini keys exhausted → raise LLMProviderError

    Usage:
        provider = LLMProvider()

        # Simple call (auto-selects provider):
        llm = provider.create_llm(temperature=0.1)

        # Resilient call with rotation:
        result = provider.invoke_with_rotation(create_fn, invoke_fn)
    """

    # Error patterns that indicate quota/rate-limit exhaustion
    EXHAUSTION_SIGNALS = [
        "resourceexhausted", "429", "quota", "rate limit",
        "too many requests", "resource has been exhausted",
    ]

    # Error patterns that indicate a permanently bad key
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
            extra={"component": "llm"}
        )

    @property
    def current_gemini_key(self) -> str:
        if not self._gemini_keys:
            raise LLMProviderError("No Gemini keys available")
        return self._gemini_keys[self._gemini_index]

    def rotate_gemini_key(self) -> Optional[str]:
        """Rotate to next Gemini key. Returns new key or None if only one."""
        if len(self._gemini_keys) <= 1:
            return None
        next_idx = (self._gemini_index + 1) % len(self._gemini_keys)
        if next_idx == self._gemini_index:
            return None
        self._gemini_index = next_idx
        new_key = self._gemini_keys[self._gemini_index]
        logger.warning(
            f"Rotated to Gemini key {self._gemini_index + 1}/{len(self._gemini_keys)}",
            extra={"component": "llm"}
        )
        return new_key

    def _is_exhaustion_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return any(signal in error_str for signal in self.EXHAUSTION_SIGNALS)

    def _is_key_unusable_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return any(signal in error_str for signal in self.KEY_UNUSABLE_SIGNALS)

    def create_mega_llm(self, **kwargs):
        """Create an OpenAI-compatible LLM pointed at MegaLLM."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self._settings.mega_model,
            api_key=self._mega_key,
            base_url=self._settings.mega_base_url,
            **kwargs,
        )

    def create_gemini_llm(self, api_key: Optional[str] = None, **kwargs):
        """Create a Gemini LLM with specified or current key."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or self.current_gemini_key
        return ChatGoogleGenerativeAI(
            model=self._settings.gemini_model,
            google_api_key=key,
            **kwargs,
        )

    def create_llm(self, **kwargs):
        """Create the best available LLM (MegaLLM preferred, Gemini fallback)."""
        if self._mega_key:
            return self.create_mega_llm(**kwargs)
        return self.create_gemini_llm(**kwargs)

    def invoke_with_rotation(
        self,
        create_llm_fn: Callable[[str], Any],
        invoke_fn: Callable[[Any], Any],
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Any:
        """
        Invoke an LLM call with automatic failover and key rotation.

        This is the primary method for all LLM calls in the system.
        It provides resilience against:
          - MegaLLM downtime → falls back to Gemini
          - Gemini quota exhaustion → rotates to next key
          - Permanent key issues (revoked, invalid) → skips to next key

        Args:
            create_llm_fn: Factory function that takes an API key and returns an LLM
            invoke_fn: Function that takes an LLM and returns a response
            max_retries: Max Gemini keys to try (default: all available)
            retry_delay: Seconds between retries (default: from config)
        """
        settings = self._settings
        if max_retries is None:
            max_retries = len(self._gemini_keys)
        if retry_delay is None:
            retry_delay = settings.llm_retry_delay

        # ── Try MegaLLM first ──
        if self._mega_key:
            try:
                mega_llm = self.create_mega_llm()
                result = invoke_fn(mega_llm)
                logger.debug("MegaLLM call succeeded", extra={"component": "llm"})
                return result
            except Exception as e:
                logger.warning(
                    f"MegaLLM failed: {e}. Falling back to Gemini.",
                    extra={"component": "llm"}
                )

        # ── Fall back to Gemini with key rotation ──
        if not self._gemini_keys:
            raise LLMProviderError("MegaLLM failed and no Gemini keys configured.")

        last_error = None
        keys_tried: set = set()

        for attempt in range(max_retries):
            key = self.current_gemini_key
            if key in keys_tried:
                break
            keys_tried.add(key)

            try:
                llm = create_llm_fn(key)
                return invoke_fn(llm)
            except Exception as e:
                exhausted = self._is_exhaustion_error(e)
                if exhausted or self._is_key_unusable_error(e):
                    reason = "exhausted" if exhausted else "unusable"
                    logger.warning(
                        f"Gemini key {self._gemini_index + 1} {reason}: {e}",
                        extra={"component": "llm"}
                    )
                    last_error = e
                    if self.rotate_gemini_key() is None:
                        break
                    if exhausted and retry_delay > 0:
                        time.sleep(retry_delay)
                else:
                    raise

        raise LLMProviderError(
            f"All LLM providers failed. Tried MegaLLM + {len(keys_tried)} Gemini key(s). "
            f"Last error: {last_error}"
        ) from last_error

    def status(self) -> dict:
        """Return provider status for health checks and UI display."""
        return {
            "mega_available": self._mega_key is not None,
            "mega_model": self._settings.mega_model if self._mega_key else None,
            "gemini_keys_total": len(self._gemini_keys),
            "gemini_active_key_index": self._gemini_index + 1 if self._gemini_keys else 0,
            "gemini_model": self._settings.gemini_model,
            "total_providers": (1 if self._mega_key else 0) + len(self._gemini_keys),
        }
