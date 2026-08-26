"""Thin Ollama client (deck: "self-hosted, open-weight model served by Ollama").

Talks to the local Ollama HTTP API using only the standard library, so the
package has no hard dependency on a running model.

`available()` lets callers fall back to deterministic heuristics when Ollama
is not reachable, which keeps the whole pipeline runnable and testable offline.
"""

import json
import urllib.error
import urllib.request
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        """Return True if an Ollama server responds on the configured host."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET",
            )

            with urllib.request.urlopen(
                req,
                timeout=min(self.timeout, 3.0),
            ) as resp:
                return resp.status == 200

        except (urllib.error.URLError, OSError, ValueError):
            return False

    def generate_json(
        self,
        system: str,
        prompt: str,
    ) -> Optional[dict]:
        """Ask the model for a JSON object.

        Returns None on any connection, HTTP, parsing, or response-format
        failure. Callers treat None as "LLM unavailable/invalid -> use fallback".
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(
                req,
                timeout=self.timeout,
            ) as resp:

                if resp.status != 200:
                    return None

                body = json.loads(
                    resp.read().decode("utf-8")
                )

            response = body.get("response")

            if not isinstance(response, str) or not response.strip():
                return None

            result = json.loads(response)

            if not isinstance(result, dict):
                return None

            return result

        except (
            urllib.error.URLError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None