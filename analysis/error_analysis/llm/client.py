"""Generic OpenAI-compatible chat-completions client.

Used as a FIXED-PROMPT tagging tool for the qualitative error analysis (see
`analysis/error_analysis/`), never as an agent, never as a source of
correctness numbers (the strict math grader in `re_polar/core/grader.py` owns
those).

Points at ANY OpenAI-compatible chat-completions endpoint, a self-hosted
vLLM/TGI/etc. server, or a hosted API, via `LLM_BASE_URL`/`LLM_MODEL`/
`LLM_API_KEY` environment variables or the constructor/call-site arguments
below. No endpoint is hardcoded.

The paper's own error-analysis results were produced against
`cyankiwi/MiniMax-M2.7-AWQ-4bit` (Hugging Face), served this way. Point this
client at your own deployment of that same checkpoint to get directly
comparable numbers, or at any other chat model to redo the methodology with
a different judge (the exact prompts are in `analysis/error_analysis/`).
"""

import asyncio
import os
import re

import httpx

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Generous default: some self-hosted endpoints under load take minutes per
# call. Tune down for a fast local server.
HTTPX_TIMEOUT = httpx.Timeout(2000.0, connect=30.0)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class LLMCallTimeout(Exception):
    """Raised by achat() when timeout_s elapses with no response.

    Enforced via asyncio.wait_for (elapsed wall-clock time since the call
    started), not httpx's own read timeout (idle time since the last byte)
    -- a server that trickles occasional bytes without ever completing the
    response can reset an idle-since-last-byte timeout indefinitely, so
    wall-clock is the only timeout that's guaranteed to actually fire.
    """


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL):
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.base_url = base_url
        self._client = httpx.Client(timeout=HTTPX_TIMEOUT)

    def _headers(self) -> dict:
        if not self.api_key:
            return {"Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple synchronous call, no enforced deadline beyond httpx's own
        (unreliable, see LLMCallTimeout) read timeout -- fine for one-off/
        interactive use, not for an unattended batch run (use achat() there
        instead)."""
        r = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def achat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout_s: float | None = None,
    ) -> str:
        """Async counterpart with a genuinely enforced wall-clock deadline
        (see LLMCallTimeout's docstring). A fresh, unpooled AsyncClient per
        call -- connection setup overhead is negligible next to a chat
        completion's typical response time, and it sidesteps any question
        of a cancelled request leaving a shared pool's connection in an
        inconsistent state. Raises LLMCallTimeout on timeout instead of
        asyncio.TimeoutError so callers don't need to import asyncio.

        Also translates a 5xx HTTP response to LLMCallTimeout: a 502/503/504
        is semantically the same "no real answer in time" story as a
        transport-level timeout, just signaled via HTTP status instead --
        scoped to 5xx specifically (not all non-2xx) so a genuine 4xx client
        bug still crashes loudly rather than being silently retried."""
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            coro = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            try:
                r = await asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else await coro
            except (asyncio.TimeoutError, httpx.TimeoutException):
                raise LLMCallTimeout(f"no response within {timeout_s}s")
            if r.status_code >= 500:
                raise LLMCallTimeout(f"server error {r.status_code} {r.reason_phrase}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
