"""Thin wrapper over the local Ollama HTTP API.

No SDK dependency — just requests against http://localhost:11434 (or
$OLLAMA_HOST). Kept deliberately small: list models, check capabilities,
and run a single generate call with optional images (base64) for vision
stages.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from . import config


class OllamaError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    name: str
    size_bytes: int = 0
    capabilities: list[str] = field(default_factory=list)
    parameter_size: str = ""
    quantization: str = ""

    @property
    def is_vision(self) -> bool:
        if self.capabilities:
            return "vision" in self.capabilities
        lowered = self.name.lower()
        return any(hint in lowered for hint in config.VISION_NAME_HINTS)


@dataclass
class GenerateResult:
    model: str
    prompt_tokens: int
    output_text: str
    latency_ms: float
    raw_response: dict[str, Any]


def is_reachable(timeout: float = 2.0) -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def list_models() -> list[ModelInfo]:
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        tags_body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError(f"Could not list models from {config.OLLAMA_HOST}: {exc}") from exc
    models = []
    for entry in tags_body.get("models", []):
        name = entry.get("name") or entry.get("model")
        details = entry.get("details", {}) or {}
        models.append(
            ModelInfo(
                name=name,
                size_bytes=entry.get("size", 0),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
            )
        )
    # Enrich with capabilities where /api/show supports it; best-effort.
    for model in models:
        try:
            model.capabilities = _fetch_capabilities(model.name)
        except OllamaError:
            pass
    return models


def _fetch_capabilities(model_name: str) -> list[str]:
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/show", json={"name": model_name}, timeout=10
        )
    except requests.RequestException as exc:
        raise OllamaError(f"api/show unreachable for {model_name}: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"api/show failed for {model_name}: {resp.status_code}")
    try:
        return resp.json().get("capabilities", []) or []
    except ValueError as exc:
        raise OllamaError(f"api/show returned a non-JSON response for {model_name}: {exc}") from exc


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    images: Optional[list[Path]] = None,
    response_format: Any = "json",
    num_predict: int = 2048,
    num_ctx: Optional[int] = None,
    think: Optional[bool] = None,
    timeout: float = 300.0,
) -> GenerateResult:
    """Run one blocking generation call against Ollama's /api/generate.

    response_format:
      - "json"  -> loose JSON mode (model free to choose shape)
      - dict    -> full JSON Schema, constrains decoding to that shape
                   (Ollama structured outputs, needs a recent Ollama build)
      - None    -> no constraint, freeform text

    num_ctx: context window size. Left unset, Ollama auto-sizes it from
    available VRAM, which can end up smaller than a call's actual prompt
    (seen in practice: a 4096-token auto default rejecting a 4389-token
    image-analysis prompt with a 400). Pass an explicit value for any call
    whose prompt can run long.
    """
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if seed is not None:
        options["seed"] = seed

    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": options,
    }
    if response_format is not None:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think
    if images:
        try:
            payload["images"] = [_encode_image(p) for p in images]
        except OSError as exc:
            # e.g. a OneDrive-dehydrated file that failed to download on read
            raise OllamaError(f"Could not read image file(s) for this call: {exc}") from exc

    started = time.perf_counter()
    try:
        resp = requests.post(f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaError(f"Could not reach Ollama at {config.OLLAMA_HOST}: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    if resp.status_code != 200:
        raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:500]}")

    try:
        body = resp.json()
    except ValueError as exc:  # covers requests'/simplejson's JSONDecodeError
        raise OllamaError(f"Ollama returned a non-JSON response: {exc} (body: {resp.text[:300]!r})") from exc
    return GenerateResult(
        model=model,
        prompt_tokens=body.get("prompt_eval_count", 0),
        output_text=body.get("response", ""),
        latency_ms=latency_ms,
        raw_response=body,
    )


def pull_model(model: str):
    """Streaming generator yielding progress dicts from /api/pull."""
    with requests.post(
        f"{config.OLLAMA_HOST}/api/pull", json={"name": model, "stream": True},
        stream=True, timeout=None,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            import json as _json
            yield _json.loads(line)
