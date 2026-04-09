"""
LLM Key Manager with MegaLLM-first strategy and Gemini fallback.

Priority:
  1. MegaLLM (MEGA_API_KEY) — model: gemini-3-pro-preview
     Base URL: https://ai.megallm.io/v1 (OpenAI-compatible)
  2. Gemini (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... or GOOGLE_API_KEY)
     with automatic key rotation on quota exhaustion

Loads keys from environment variables:
  MEGA_API_KEY
  GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
  (falls back to GOOGLE_API_KEY if numbered keys not found)
"""

import os
import time
import logging
from typing import List, Optional, Callable, Any

logger = logging.getLogger(__name__)

MEGA_BASE_URL = "https://ai.megallm.io/v1"
MEGA_MODEL = "gemini-3-pro-preview"


class GeminiKeyManager:
    """
    LLM provider manager: tries MegaLLM first, falls back to Gemini.

    Usage:
        key_manager = GeminiKeyManager()
        llm = key_manager.create_llm(model="gemini-2.5-flash", temperature=0.1)
        # Keys and providers rotate automatically on quota errors
    """

    # Quota / rate-limit error signals
    EXHAUSTION_SIGNALS = [
        "resourceexhausted",
        "429",
        "quota",
        "rate limit",
        "too many requests",
        "resource has been exhausted",
    ]

    def __init__(self):
        self._mega_key: Optional[str] = self._load_mega_key()
        self._gemini_keys: List[str] = self._load_gemini_keys()
        self._gemini_index: int = 0

        if not self._mega_key and not self._gemini_keys:
            raise EnvironmentError(
                "No API keys found. Set MEGA_API_KEY or GOOGLE_API_KEY / "
                "GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... in your .env file."
            )

        if self._mega_key:
            logger.info("GeminiKeyManager: MegaLLM key loaded (primary provider)")
        logger.info(
            f"GeminiKeyManager: {len(self._gemini_keys)} Gemini key(s) loaded (fallback)"
        )

    # ------------------------------------------------------------------
    # Key loading
    # ------------------------------------------------------------------

    def _load_mega_key(self) -> Optional[str]:
        key = os.getenv("MEGA_API_KEY")
        return key.strip() if key else None

    def _load_gemini_keys(self) -> List[str]:
        keys = []
        i = 1
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key:
                break
            keys.append(key.strip())
            i += 1
        if not keys:
            key = os.getenv("GOOGLE_API_KEY")
            if key:
                keys.append(key.strip())
        return keys

    # ------------------------------------------------------------------
    # Current Gemini key helpers
    # ------------------------------------------------------------------

    @property
    def current_key(self) -> str:
        """Return the currently active Gemini API key."""
        if not self._gemini_keys:
            raise RuntimeError("No Gemini API keys available.")
        return self._gemini_keys[self._gemini_index]

    def rotate(self) -> Optional[str]:
        """Rotate to the next Gemini key. Returns new key or None if only one."""
        if len(self._gemini_keys) <= 1:
            return None
        next_index = (self._gemini_index + 1) % len(self._gemini_keys)
        if next_index == self._gemini_index:
            return None
        self._gemini_index = next_index
        new_key = self._gemini_keys[self._gemini_index]
        logger.warning(
            f"Rotated to Gemini key {self._gemini_index + 1}/{len(self._gemini_keys)} "
            f"(ends ...{new_key[-6:]})"
        )
        return new_key

    def is_exhaustion_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return any(signal in error_str for signal in self.EXHAUSTION_SIGNALS)

    # ------------------------------------------------------------------
    # LLM factories
    # ------------------------------------------------------------------

    def _create_mega_llm(self, **kwargs):
        """Create an OpenAI-compatible ChatOpenAI pointed at MegaLLM."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MEGA_MODEL,
            api_key=self._mega_key,
            base_url=MEGA_BASE_URL,
            **kwargs,
        )

    def _create_gemini_llm(self, api_key: str, model: str = "gemini-2.5-flash", **kwargs):
        """Create a Gemini LLM with the given key."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            **kwargs,
        )

    def create_llm(self, model: str = "gemini-2.5-flash", **kwargs):
        """
        Return the primary LLM: MegaLLM if available, else Gemini.
        Used for simple one-shot calls; use invoke_with_rotation for resilience.
        """
        if self._mega_key:
            return self._create_mega_llm(**kwargs)
        return self._create_gemini_llm(self.current_key, model=model, **kwargs)

    # ------------------------------------------------------------------
    # Resilient invocation
    # ------------------------------------------------------------------

    def invoke_with_rotation(
        self,
        create_llm_fn: Callable[[str], Any],
        invoke_fn: Callable[[Any], Any],
        max_retries: int = None,
        retry_delay: float = 2.0,
    ) -> Any:
        """
        Invoke with MegaLLM first, then fall back to Gemini key rotation.

        The create_llm_fn / invoke_fn signature is kept for backwards
        compatibility — they are used only for the Gemini fallback path.

        Args:
            create_llm_fn: callable(api_key) -> Gemini LLM (fallback)
            invoke_fn:      callable(llm) -> result
            max_retries:    Gemini keys to try after MegaLLM fails
            retry_delay:    seconds to wait before each retry
        """
        # --- Try MegaLLM first ---
        if self._mega_key:
            try:
                mega_llm = self._create_mega_llm()
                result = invoke_fn(mega_llm)
                logger.debug("MegaLLM call succeeded.")
                return result
            except Exception as e:
                logger.warning(
                    f"MegaLLM failed ({e}). Falling back to Gemini keys..."
                )

        # --- Fall back to Gemini with key rotation ---
        if not self._gemini_keys:
            raise RuntimeError(
                "MegaLLM failed and no Gemini keys are configured as fallback."
            )

        if max_retries is None:
            max_retries = len(self._gemini_keys)

        last_error = None
        keys_tried: set = set()

        for attempt in range(max_retries):
            key = self.current_key

            if key in keys_tried:
                break
            keys_tried.add(key)

            try:
                llm = create_llm_fn(key)
                return invoke_fn(llm)

            except Exception as e:
                if self.is_exhaustion_error(e):
                    logger.warning(
                        f"Gemini key {self._gemini_index + 1} exhausted: {e}. "
                        f"Rotating to next key..."
                    )
                    last_error = e
                    rotated = self.rotate()
                    if rotated is None:
                        break
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                else:
                    raise

        raise RuntimeError(
            f"MegaLLM and all {len(self._gemini_keys)} Gemini key(s) failed. "
            f"Last error: {last_error}"
        ) from last_error

    def status(self) -> dict:
        """Return current provider/key status for display."""
        return {
            "mega_available": self._mega_key is not None,
            "mega_model": MEGA_MODEL if self._mega_key else None,
            "gemini_total_keys": len(self._gemini_keys),
            "gemini_active_key_index": self._gemini_index + 1 if self._gemini_keys else 0,
            "gemini_active_key_suffix": (
                f"...{self.current_key[-6:]}" if self._gemini_keys else "N/A"
            ),
        }
