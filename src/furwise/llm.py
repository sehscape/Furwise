"""One small client for free LLM backends.

* ``groq`` (default) or any ``openai_compatible`` free tier: chat-completions API,
  key from the environment, requests paced to the free tier's tokens-per-minute.
* ``ollama``: a local model, no key and no limits. Uses Ollama's native /api/chat
  with an explicit context size, because the OpenAI-compatible endpoint silently
  truncates long prompts to the default window.

Generator and judge are configured separately (FURWISE_GEN_*, FURWISE_JUDGE_*),
so the RAGAS judge can be a different model family from the generator.
Every reply is cached on disk by a hash of the request, so re-running an
evaluation never spends free-tier quota twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import requests

from furwise.settings import RUNS

log = logging.getLogger(__name__)

# Free tiers. Limits are per minute here; the daily caps are what make a full
# evaluation take more than one day on a single provider (see docs/evaluation.md).
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "tpm": 8000,
        "rpm": 30,
        "model": {"GEN": "qwen/qwen3.8-27b", "JUDGE": "openai/gpt-oss-120b"},
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "tpm": 0,
        "rpm": 10,
        "model": {"GEN": "gemini-2.5-flash-lite", "JUDGE": "gemini-2.5-flash"},
    },
    "openai_compatible": {
        "base_url": "",
        "key_env": "FURWISE_API_KEY",
        "tpm": 0,
        "rpm": 0,
        "model": {},
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "key_env": "",
        "tpm": 0,
        "rpm": 0,
        "model": {"GEN": "llama3.2", "JUDGE": "llama3.2"},
    },
}
CACHE_DIR = RUNS / "llm_cache"
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    tpm: int = 0
    rpm: int = 0
    reasoning_effort: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def config_from_env(role: str = "GEN") -> LLMConfig:
    def env(name: str, default: str = "") -> str:
        return os.getenv(f"FURWISE_{role}_{name}", default)

    provider = env("PROVIDER", "groq").lower()
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    preset = PROVIDERS[provider]
    key = env("API_KEY") or (os.getenv(preset["key_env"], "") if preset["key_env"] else "")
    if provider != "ollama" and not key:
        raise LLMError(f"set {preset['key_env']} in .env (a free key, see .env.example)")
    base_url = env("BASE_URL", preset["base_url"]).rstrip("/")
    if not base_url:
        raise LLMError(f"set FURWISE_{role}_BASE_URL for provider {provider}")
    model = env("MODEL", preset["model"].get(role, ""))
    if not model:
        raise LLMError(f"set FURWISE_{role}_MODEL for provider {provider}")
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=key,
        tpm=int(env("TPM", str(preset["tpm"]))),
        rpm=int(env("RPM", str(preset["rpm"]))),
        reasoning_effort=env("REASONING_EFFORT", "low" if "gpt-oss" in model else ""),
    )


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0  # model time, excluding any waiting on free-tier limits
    wait_ms: float = 0.0
    cached: bool = False


class LLMClient:
    def __init__(self, config: LLMConfig, use_cache: bool = True):
        self.config = config
        self.use_cache = use_cache
        self.session = requests.Session()
        if config.api_key:
            self.session.headers["Authorization"] = f"Bearer {config.api_key}"
        self._lock = threading.Lock()
        self._window: deque[tuple[float, int]] = deque()
        self._last = 0.0

    @classmethod
    def from_env(cls, role: str = "GEN") -> LLMClient:
        return cls(config_from_env(role))

    # -------------------------------------------------------------- pacing
    def _wait_for_budget(self, tokens: int) -> float:
        """Requests/minute spacing plus a sliding one-minute token budget: wait
        *before* a request instead of collecting 429s. Returns seconds waited."""
        waited = 0.0
        with self._lock:
            if self.config.rpm > 0:
                gap = self._last + 60.0 / self.config.rpm - time.monotonic()
                if gap > 0:
                    time.sleep(gap)
                    waited += gap
                self._last = time.monotonic()
            if self.config.tpm <= 0:
                return waited
            while True:
                now = time.monotonic()
                while self._window and now - self._window[0][0] > 60:
                    self._window.popleft()
                if not self._window or sum(t for _, t in self._window) + tokens <= self.config.tpm:
                    self._window.append((now, tokens))
                    return waited
                pause = 60 - (now - self._window[0][0]) + 0.5
                time.sleep(pause)
                waited += pause

    # -------------------------------------------------------------- cache
    def _key(self, payload: dict) -> str:
        blob = json.dumps({"m": self.config.label, "p": payload}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cached(self, key: str) -> Completion | None:
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if self.use_cache and path.exists():
            data = json.loads(path.read_text("utf-8"))
            return Completion(**{**data, "cached": True})
        return None

    def _store(self, key: str, c: Completion) -> None:
        if not self.use_cache:
            return
        path = CACHE_DIR / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in c.__dict__.items() if k != "cached"}
        path.write_text(json.dumps(data), "utf-8")

    # -------------------------------------------------------------- API
    def chat(
        self, messages: list[dict], max_tokens: int = 600, json_mode: bool = False
    ) -> Completion:
        if self.config.provider == "ollama":
            url = f"{self.config.base_url}/api/chat"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens},
            }
            if json_mode:
                payload["format"] = "json"
        else:
            url = f"{self.config.base_url}/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            if self.config.reasoning_effort:
                payload["reasoning_effort"] = self.config.reasoning_effort
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

        key = self._key(payload)
        hit = self._cached(key)
        if hit:
            return hit

        estimate = sum(len(str(m.get("content", ""))) for m in messages) // 4 + max_tokens
        waited = self._wait_for_budget(estimate)
        for attempt in range(6):
            start = time.perf_counter()
            try:
                resp = self.session.post(url, json=payload, timeout=180)
            except requests.RequestException as exc:
                if attempt == 5:
                    raise LLMError(f"{self.config.label}: {exc}") from exc
                time.sleep(2**attempt)
                waited += 2**attempt
                continue
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                delay = min(float(resp.headers.get("retry-after") or 2 ** (attempt + 1)), 90)
                if resp.status_code == 429 and "per day" in resp.text.lower():
                    raise LLMError(
                        f"{self.config.label}: daily free-tier limit reached. "
                        "Cached answers are kept; run again later to resume."
                    )
                log.warning(
                    "%s -> %d, retrying in %.0fs", self.config.label, resp.status_code, delay
                )
                time.sleep(delay)
                waited += delay
                continue
            raise LLMError(f"{self.config.label} -> {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if self.config.provider == "ollama":
            text = data.get("message", {}).get("content", "")
            usage = (data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        else:
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = data.get("usage") or {}
            usage = (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        out = Completion(_THINK.sub("", text).strip(), usage[0], usage[1], elapsed, waited * 1000)
        self._store(key, out)
        return out


def extract_json(text: str) -> str:
    """The JSON object inside a reply, tolerating ```json fences and chatter."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text
