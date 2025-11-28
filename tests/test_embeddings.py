from __future__ import annotations

import textwrap

import pytest

from tiangong_ai_workspace.secrets import OpenAICompatibleEmbeddingSecrets, Secrets, load_secrets
from tiangong_ai_workspace.tooling.embeddings import OpenAICompatibleEmbeddingClient, OpenAIEmbeddingError


def test_load_secrets_parses_embedding_configuration(tmp_path) -> None:
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        textwrap.dedent(
            """
            [openai_compatitble_embedding]
            url = "http://localhost:9000/v1/"
            model = "test-embedding"
            """
        ),
        encoding="utf-8",
    )

    secrets = load_secrets(path=secrets_file)
    assert secrets.openai_compatible_embedding is not None
    assert secrets.openai_compatible_embedding.url == "http://localhost:9000/v1"
    assert secrets.openai_compatible_embedding.model == "test-embedding"


def test_openai_embedding_client_invokes_service(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OpenAICompatibleEmbeddingSecrets(url="http://localhost:8004/v1", model="test-embedding")
    secrets = Secrets(openai=None, mcp_servers={}, openai_compatible_embedding=config)
    client = OpenAICompatibleEmbeddingClient(secrets=secrets)

    captured: dict[str, object] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "model": "test-embedding",
                "data": [
                    {
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                        "index": 0,
                        "object": "embedding",
                    }
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }

    def fake_post(self, url: str, *, headers, json):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _StubResponse()

    monkeypatch.setattr(OpenAICompatibleEmbeddingClient, "_post", fake_post, raising=False)

    result = client.embed(["hello world"])

    assert captured["url"].endswith("/embeddings")
    assert captured["json"]["input"] == ["hello world"]
    assert result.embeddings[0][:2] == [0.1, 0.2]
    assert result.dimensions == 4
    assert result.model == "test-embedding"
    assert result.warnings == ()


def test_openai_embedding_client_requires_configuration() -> None:
    secrets = Secrets(openai=None, mcp_servers={}, openai_compatible_embedding=None)
    with pytest.raises(OpenAIEmbeddingError):
        OpenAICompatibleEmbeddingClient(secrets=secrets)
