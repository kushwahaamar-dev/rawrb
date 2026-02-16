"""Unified LLM client for Ollama with structured output + retry.

Adapted from cg-cot/src/llm_client.py with enhancements for
multi-model benchmarking.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import MAX_TOKENS, OLLAMA_BASE_URL, TEMPERATURE

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _model_to_example_dict(model_cls: type[BaseModel]) -> dict:
    """Build a concise example dict from a Pydantic model for the prompt."""
    import typing

    example = {}
    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())

        if origin is list or origin is typing.List:
            inner = args[0] if args else str
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                example[name] = [_model_to_example_dict(inner)]
            else:
                example[name] = [f"<{name}_item>"]
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            example[name] = _model_to_example_dict(annotation)
        elif annotation is str:
            example[name] = f"<{name}>"
        elif annotation is int:
            example[name] = 0
        elif annotation is float:
            example[name] = 0.0
        elif annotation is bool:
            example[name] = True
        else:
            default = field_info.default
            if default is not None and default is not ...:
                example[name] = default
            else:
                example[name] = f"<{name}>"
    return example


def _schema_to_example(model_cls: type[BaseModel]) -> str:
    """Build a JSON example string from a Pydantic model."""
    return json.dumps(_model_to_example_dict(model_cls), indent=2)


# ── Ollama Client ───────────────────────────────────────────────────

class OllamaClient:
    """Local Ollama models via REST API with usage tracking."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ):
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Tracking
        self.total_tokens_used: int = 0
        self.call_count: int = 0

    @retry(
        retry=retry_if_exception_type(
            (ValidationError, json.JSONDecodeError, ConnectionError,
             requests.RequestException)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """Structured JSON call → parsed into *response_model*."""
        example = _schema_to_example(response_model)
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"Respond ONLY with valid JSON. Example format:\n{example}"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        start = time.time()
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["message"]["content"]
        self._track(data, time.time() - start)
        return response_model.model_validate_json(raw)

    @retry(
        retry=retry_if_exception_type(
            (ConnectionError, requests.RequestException)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def call_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Free-form text call (no JSON parsing)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.time()
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        self._track(data, time.time() - start)
        return data["message"]["content"]

    def _track(self, data: dict, elapsed: float) -> None:
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        self.total_tokens_used += tokens
        self.call_count += 1
        logger.debug(
            "Ollama [%s] call #%d  %.2fs  tokens=%d",
            self.model_name, self.call_count, elapsed, tokens,
        )

    def reset_tracking(self) -> None:
        self.total_tokens_used = 0
        self.call_count = 0

    @property
    def usage_summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_tokens": self.total_tokens_used,
            "api_calls": self.call_count,
        }


# ── Factory ─────────────────────────────────────────────────────────

LLMClient = OllamaClient


def make_client(model: str, **kwargs) -> LLMClient:
    """Create an Ollama client for the given model name."""
    return OllamaClient(model=model, **kwargs)
