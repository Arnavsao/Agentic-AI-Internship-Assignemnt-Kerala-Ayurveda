"""
Gemini API Key Manager with automatic rotation.

Loads keys from environment variables:
  GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
  (falls back to GOOGLE_API_KEY if numbered keys not found)

When a key hits a quota/rate-limit error (ResourceExhausted / 429),
the manager rotates to the next available key transparently.
"""

import os
import time
import logging
from typing import List, Optional, Callable, Any

logger = logging.getLogger(__name__)


class GeminiKeyManager:
    """
    Thread-safe Gemini API key pool with automatic rotation on exhaustion.

    Usage:
        key_manager = GeminiKeyManager()
        llm = key_manager.create_llm(model="gemini-2.5-flash", temperature=0.1)
        # Use llm normally — keys rotate automatically on quota errors
    """

    # Google API quota / rate-limit error signals
    EXHAUSTION_SIGNALS = [
        "resourceexhausted",
        "429",
        "quota",
        "rate limit",
        "too many requests",
        "resource has been exhausted",
    ]

    def __init__(self):
        self.keys: List[str] = self._load_keys()
        self._index: int = 0

        if not self.keys:
            raise EnvironmentError(
                "No Gemini API keys found. Set GOOGLE_API_KEY or "
                "GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... in your .env file."
            )

        logger.info(f"GeminiKeyManager: loaded {len(self.keys)} key(s)")

    def _load_keys(self) -> List[str]:
        """Load all available API keys from environment variables."""
        keys = []

        # Try numbered keys first: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
        i = 1
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key:
                break
            keys.append(key.strip())
            i += 1

        # Fallback to plain GOOGLE_API_KEY
        if not keys:
            key = os.getenv("GOOGLE_API_KEY")
            if key:
                keys.append(key.strip())

        return keys

    @property
    def current_key(self) -> str:
        """Return the currently active API key."""
        return self.keys[self._index]

    def rotate(self) -> Optional[str]:
        """
        Rotate to the next key in the pool.
        Returns the new key, or None if all keys are exhausted.
        """
        next_index = (self._index + 1) % len(self.keys)
        if next_index == self._index:
            return None  # Only one key, can't rotate further
        self._index = next_index
        new_key = self.keys[self._index]
        logger.warning(
            f"Rotated to API key {self._index + 1}/{len(self.keys)} "
            f"(ends ...{new_key[-6:]})"
        )
        return new_key

    def is_exhaustion_error(self, error: Exception) -> bool:
        """Check whether an exception is a quota/rate-limit error."""
        error_str = str(error).lower()
        return any(signal in error_str for signal in self.EXHAUSTION_SIGNALS)

    def invoke_with_rotation(
        self,
        create_llm_fn: Callable[[str], Any],
        invoke_fn: Callable[[Any], Any],
        max_retries: int = None,
        retry_delay: float = 2.0,
    ) -> Any:
        """
        Invoke an LLM call with automatic key rotation on exhaustion.

        Args:
            create_llm_fn: callable(api_key) -> LLM instance
            invoke_fn:      callable(llm) -> result
            max_retries:    how many keys to try (default: all keys)
            retry_delay:    seconds to wait before retry

        Returns:
            The result from invoke_fn on the first successful key.

        Raises:
            The last exception if all keys are exhausted.
        """
        if max_retries is None:
            max_retries = len(self.keys)

        last_error = None
        keys_tried = set()

        for attempt in range(max_retries):
            key = self.current_key

            if key in keys_tried:
                # We've looped around — all keys exhausted
                break
            keys_tried.add(key)

            try:
                llm = create_llm_fn(key)
                return invoke_fn(llm)

            except Exception as e:
                if self.is_exhaustion_error(e):
                    logger.warning(
                        f"Key {self._index + 1} exhausted: {e}. "
                        f"Rotating to next key..."
                    )
                    last_error = e
                    rotated = self.rotate()
                    if rotated is None:
                        break
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                else:
                    raise  # Non-quota errors propagate immediately

        raise RuntimeError(
            f"All {len(self.keys)} Gemini API key(s) exhausted or failed. "
            f"Last error: {last_error}"
        ) from last_error

    def create_llm(self, model: str = "gemini-2.5-flash", **kwargs):
        """
        Create a ChatGoogleGenerativeAI instance with the current key.
        Re-call this after rotation to get an LLM with the new key.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.current_key,
            **kwargs
        )

    def status(self) -> dict:
        """Return current key pool status for display."""
        return {
            "total_keys": len(self.keys),
            "active_key_index": self._index + 1,
            "active_key_suffix": f"...{self.current_key[-6:]}",
        }
