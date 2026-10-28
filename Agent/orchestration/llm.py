"""Minimal Ollama chat client (standard library only) and the interface the agents depend on.

``chat`` returns ``{"content": str, "tool_calls": [{"name": str, "arguments": dict}]}``. Tests
and offline runs use a fake with the same method, so no agent code imports Ollama directly.
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b", temperature: float = 0.0,
                 seed: int = 42, timeout_s: float = 90.0):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.temperature, self.seed, self.timeout_s = temperature, seed, timeout_s

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"Ollama request failed: {e}")

    def available(self) -> Tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=3) as r:
                names = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
        except Exception as e:
            return False, f"Ollama not reachable at {self.base_url} ({type(e).__name__})"
        if not any(n == self.model or n.split(":")[0] == self.model.split(":")[0] and ":" not in self.model for n in names):
            return False, f"Model '{self.model}' is not pulled (run: ollama pull {self.model}); available: {names[:5]}"
        return True, f"Ollama ready ({self.model})"

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
             format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "options": {"temperature": self.temperature, "seed": self.seed}}
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format
        msg = self._post("/api/chat", payload).get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append({"name": fn.get("name", ""), "arguments": args if isinstance(args, dict) else {}})
        return {"content": msg.get("content", "") or "", "tool_calls": calls}
