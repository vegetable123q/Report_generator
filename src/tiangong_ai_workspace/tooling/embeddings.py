"""OpenAI-compatible embedding utilities for the workspace."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import httpx

from ..secrets import OpenAICompatibleEmbeddingSecrets, Secrets, load_secrets

LOGGER = logging.getLogger(__name__)

__all__ = ["EmbeddingResult", "OpenAICompatibleEmbeddingClient", "OpenAIEmbeddingError"]


class OpenAIEmbeddingError(RuntimeError):
    """Raised when the OpenAI-compatible embedding request fails."""


@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    """Structured embedding response payload."""

    embeddings: list[list[float]]
    model: str
    dimensions: Optional[int]
    usage: Mapping[str, Any] | None
    raw_response: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class OpenAICompatibleEmbeddingClient:
    """Lightweight client for OpenAI-compatible embedding services."""

    secrets: Optional[Secrets] = None
    config: Optional[OpenAICompatibleEmbeddingSecrets] = None
    timeout: float = 30.0
    http_client: Optional[httpx.Client] = None
    _config: OpenAICompatibleEmbeddingSecrets = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = self.config or loaded.openai_compatible_embedding
        if config is None:
            raise OpenAIEmbeddingError("OpenAI-compatible embedding secrets are not configured.")
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", config)

    def embed(
        self,
        inputs: Sequence[str],
        *,
        model_override: str | None = None,
        encoding_format: str = "float",
        user: str | None = None,
    ) -> EmbeddingResult:
        """Generate embeddings for one or more inputs."""

        prepared_inputs = [text for text in (value.strip() for value in inputs) if text]
        if not prepared_inputs:
            raise OpenAIEmbeddingError("At least one non-empty input text is required to generate embeddings.")

        model_name = (model_override or self._config.model).strip()
        if not model_name:
            raise OpenAIEmbeddingError("Embedding model name cannot be empty.")

        payload: MutableMapping[str, Any] = {
            "model": model_name,
            "input": list(prepared_inputs),
            "encoding_format": encoding_format,
        }
        if user:
            payload["user"] = user

        url = f"{self._config.url.rstrip('/')}/embeddings"
        headers = _build_headers(self._config.api_key)

        LOGGER.debug("Calling embedding endpoint %s", url)
        data = self._call_embedding_api(url, headers, payload)

        embeddings, actual_dim = _parse_embeddings(data)
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else None
        model = str(data.get("model") or model_name)

        return EmbeddingResult(
            embeddings=embeddings,
            model=model,
            dimensions=actual_dim,
            usage=usage,
            raw_response=data,
        )

    def _post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=json, timeout=self.timeout)
        return httpx.post(url, headers=headers, json=json, timeout=self.timeout)

    def _call_embedding_api(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            response = self._post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.exception("Embedding request failed")
            _raise_embedding_http_error(exc)
        return _parse_response_json(response)


def _build_headers(api_key: Optional[str]) -> Mapping[str, str]:
    headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _parse_response_json(response: httpx.Response) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise OpenAIEmbeddingError("Embedding service returned invalid JSON.") from exc
    if not isinstance(data, Mapping):
        raise OpenAIEmbeddingError("Embedding service returned an unexpected payload.")
    return data


def _parse_embeddings(payload: Mapping[str, Any]) -> tuple[list[list[float]], Optional[int]]:
    data_entries = payload.get("data")
    if not isinstance(data_entries, Sequence) or isinstance(data_entries, (str, bytes)):
        raise OpenAIEmbeddingError("Embedding service did not return a valid 'data' list.")

    embeddings: list[list[float]] = []
    actual_dim: Optional[int] = None
    for entry in data_entries:
        if not isinstance(entry, Mapping):
            raise OpenAIEmbeddingError("Embedding record must be a JSON object.")
        vector = entry.get("embedding")
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise OpenAIEmbeddingError("Embedding vector must be a sequence of numbers.")
        float_vector = [float(value) for value in vector]
        if actual_dim is None:
            actual_dim = len(float_vector)
        elif len(float_vector) != actual_dim:
            raise OpenAIEmbeddingError("Embedding vectors have inconsistent dimensions.")
        embeddings.append(float_vector)

    if not embeddings:
        raise OpenAIEmbeddingError("Embedding service returned an empty result set.")

    return embeddings, actual_dim


def _raise_embedding_http_error(exc: httpx.HTTPError) -> None:
    message = _format_http_error(exc)
    raise OpenAIEmbeddingError(message) from exc


def _format_http_error(exc: httpx.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return f"HTTP error generating embeddings: {exc}"
    detail = _extract_error_detail(response)
    status = getattr(response, "status_code", None)
    if status is not None:
        prefix = f"HTTP {status}"
    else:
        prefix = "HTTP error"
    if detail:
        return f"{prefix} generating embeddings: {detail}"
    return f"{prefix} generating embeddings: {exc}"


def _extract_error_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return response.text or None
    if isinstance(payload, Mapping):
        if "error" in payload and isinstance(payload["error"], Mapping):
            message = payload["error"].get("message")
            if message:
                return str(message)
        message = payload.get("message")
        if message:
            return str(message)
    return response.text or None
