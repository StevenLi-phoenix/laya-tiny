"""OpenAI-compatible chat client for the "big model" steps (task design, synthetic data).

Every raw HTTP response is written to disk *before* it is parsed, keyed by a hash of the request,
so a re-run (or a crash in parsing) never pays for the same generation twice. LLM output is
untrusted: callers get parsed JSON only after `extract_json` and must validate it themselves.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .util import log, sha256_bytes, stable_json

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in `text` (tolerates code fences and leading chatter)."""
    if not isinstance(text, str) or not text.strip():
        raise LLMError("empty LLM content")
    body = _FENCE.sub("", text.strip())
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        raise LLMError(f"no JSON in LLM content: {text[:120]!r}")
    start = min(starts)
    end = max(body.rfind("}"), body.rfind("]"))
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"unparseable JSON from LLM: {e}; head={text[:120]!r}") from e


class LLM:
    def __init__(self, cfg: dict[str, Any], cache_dir: Path) -> None:
        self.cfg = cfg
        self.base_url = os.environ.get("LAYA_TINY_LLM_URL") or cfg["base_url"]
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0

    def _key(self, payload: dict[str, Any]) -> str:
        return sha256_bytes(stable_json({"url": self.base_url, "payload": payload}).encode())[:24]

    def chat(self, messages: list[dict[str, str]], *, seed: int = 0, temperature: float | None = None,
             max_tokens: int | None = None, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.get("model", "local"),
            "messages": messages,
            "temperature": float(self.cfg.get("temperature", 0.8) if temperature is None else temperature),
            "max_tokens": int(max_tokens or self.cfg.get("max_tokens", 4000)),
            "seed": int(seed),
            **(self.cfg.get("extra_body") or {}),
        }
        if json_mode and self.cfg.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        path = self.cache_dir / f"{self._key(payload)}.json"
        if path.exists():
            self.cache_hits += 1
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = self._post(payload)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": payload, "response": raw}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)                          # raw response on disk before any parsing
            raw = {"request": payload, "response": raw}
        try:
            msg = raw["response"]["choices"][0]["message"]
            content = msg.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed LLM response in {path.name}: {e}") from e
        if raw["response"]["choices"][0].get("finish_reason") == "length":
            log.warning("LLM hit max_tokens (%s); output may be truncated", payload["max_tokens"])
        return content

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        key_env = self.cfg.get("api_key_env")
        if key_env and os.environ.get(key_env):
            headers["Authorization"] = "Bearer " + os.environ[key_env]   # value never logged
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers, method="POST")
        retries = int(self.cfg.get("retries", 3))
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout", 600))) as r:  # noqa: S310
                    body = json.loads(r.read().decode("utf-8"))
                self.calls += 1
                usage = body.get("usage") or {}
                log.debug("llm call %.1fs, %s completion tokens", time.perf_counter() - t0,
                          usage.get("completion_tokens"))
                return body
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                if attempt == retries:
                    raise LLMError(f"LLM endpoint {self.base_url} failed after {retries + 1} tries: {e}") from e
                wait = 5 * (attempt + 1)
                log.warning("LLM call failed (%s), retry in %ds", e, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def chat_json(self, messages: list[dict[str, str]], validate: Callable[[Any], Any], *, seed: int = 0,
                  attempts: int = 3, **kw: Any) -> Any:
        """Ask for JSON, validate it, and on failure retry with the validation error fed back."""
        msgs = list(messages)
        last: Exception | None = None
        for i in range(attempts):
            content = self.chat(msgs, seed=seed + 1000 * i, **kw)
            try:
                return validate(extract_json(content))
            except (LLMError, ValueError, TypeError, KeyError) as e:
                last = e
                log.warning("LLM output rejected (attempt %d/%d): %s", i + 1, attempts, str(e)[:200])
                msgs = list(messages) + [{"role": "assistant", "content": content[:4000]},
                                         {"role": "user", "content": f"That was invalid: {e}. Reply with corrected JSON only."}]
        raise LLMError(f"LLM never produced valid output: {last}")


def make_llm(cfg: dict[str, Any], cache_dir: Path) -> Any:
    if cfg.get("kind", "openai") == "fake":
        from .fake_llm import FakeLLM

        return FakeLLM(cfg, cache_dir)
    return LLM(cfg, cache_dir)
