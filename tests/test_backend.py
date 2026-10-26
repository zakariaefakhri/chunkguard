import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from chunkguard import backend
from chunkguard.backend import Settings, _batches, async_chunker_from_env, chunker_from_env

VOCABULARY = ["resort", "meer", "hund"]
DOCUMENT = "Das Resort liegt am Meer. Ein Hund bellt. Das Meer ist warm im Resort."


def vector(text: str) -> list[float]:
    return [float(text.lower().count(word)) for word in VOCABULARY]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for key in Settings.model_fields:
        monkeypatch.delenv(f"CHUNKGUARD_{key.upper()}", raising=False)

    def write(text: str) -> str:
        path = tmp_path / ".env"
        path.write_text(text)
        return str(path)

    return write


class Recorder:
    """Stands in for litellm and keeps the keyword arguments of every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def completion(self, **kwargs):
        self.calls.append(("completion", kwargs))
        message = SimpleNamespace(content="Resort am Meer")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def embedding(self, **kwargs):
        self.calls.append(("embedding", kwargs))
        items = [{"index": i, "embedding": vector(t)} for i, t in enumerate(kwargs["input"])]
        return SimpleNamespace(data=items[::-1])  # providers may return any order

    async def acompletion(self, **kwargs):
        return self.completion(**kwargs)

    async def aembedding(self, **kwargs):
        return self.embedding(**kwargs)


@pytest.fixture
def recorder(monkeypatch):
    fake = Recorder()
    monkeypatch.setattr(backend, "litellm", fake)
    return fake


def test_from_env_passes_models_and_connection(env_file, recorder):
    path = env_file(
        "CHUNKGUARD_LLM_MODEL=azure/gpt-deploy\n"
        "CHUNKGUARD_LLM_API_KEY=k1\n"
        "CHUNKGUARD_LLM_API_BASE=https://x.openai.azure.com\n"
        "CHUNKGUARD_LLM_API_VERSION=2024-10-21\n"
        "CHUNKGUARD_EMBEDDING_MODEL=ollama/bge-m3\n"
        "CHUNKGUARD_EMBEDDING_API_BASE=http://localhost:11434\n"
        "CHUNKGUARD_EMBEDDING_API_KEY=\n"
        "CHUNKGUARD_MIN_CHUNK_CHARS=0\n"
    )
    chunks = chunker_from_env(path).split("d1", DOCUMENT, "de")

    (_, llm), (_, embedding) = recorder.calls
    assert llm["model"] == "azure/gpt-deploy"
    assert (llm["api_key"], llm["api_base"], llm["api_version"]) == (
        "k1",
        "https://x.openai.azure.com",
        "2024-10-21",
    )
    assert DOCUMENT in llm["messages"][0]["content"]
    assert (llm["timeout"], llm["num_retries"]) == (120, 2)
    # Unset settings are not passed, so LiteLLM falls back to provider variables.
    assert embedding["model"] == "ollama/bge-m3"
    assert embedding["api_base"] == "http://localhost:11434"
    assert "api_key" not in embedding and "api_version" not in embedding
    assert (embedding["timeout"], embedding["num_retries"]) == (120, 2)
    assert [c.relevant for c in chunks] == [True, False, True]


def test_async_from_env(env_file, recorder):
    path = env_file(
        "CHUNKGUARD_LLM_MODEL=anthropic/m\nCHUNKGUARD_EMBEDDING_MODEL=gemini/e\n"
        "CHUNKGUARD_MIN_CHUNK_CHARS=0\n"
    )
    chunks = asyncio.run(async_chunker_from_env(path).split("d1", DOCUMENT, "de"))
    assert [name for name, _ in recorder.calls] == ["completion", "embedding"]
    assert [c.relevant for c in chunks] == [True, False, True]


def test_zero_switches_limits_off_and_empty_counts_as_unset(env_file):
    path = env_file(
        "CHUNKGUARD_LLM_MODEL=m\nCHUNKGUARD_EMBEDDING_MODEL=e\nCHUNKGUARD_LLM_API_BASE=\n"
        "CHUNKGUARD_MAX_CHUNK_CHARS=0\nCHUNKGUARD_MIN_CHUNK_CHARS=0\n"
        "CHUNKGUARD_SUMMARY_MAX_CHARS=0\nCHUNKGUARD_EMBEDDING_BATCH_SIZE=\n"
        "CHUNKGUARD_REQUEST_TIMEOUT=30\nCHUNKGUARD_MAX_RETRIES=0\n"
    )
    assert Settings(_env_file=path).llm_api_base is None
    assert Settings(_env_file=path).embedding_batch_size == 64
    assert (Settings(_env_file=path).request_timeout, Settings(_env_file=path).max_retries) == (
        30,
        0,
    )
    for chunker in (chunker_from_env(path), async_chunker_from_env(path)):
        assert (chunker.max_length, chunker.min_length, chunker.summary_max_chars) == (
            None,
            0,
            None,
        )
        assert chunker.length is len


def test_missing_model_names_the_variables(env_file):
    with pytest.raises(ValueError, match=r"CHUNKGUARD_EMBEDDING_MODEL.*set them in"):
        chunker_from_env(env_file("CHUNKGUARD_LLM_MODEL=m\n"))


def test_embedding_batches_respect_count_and_size_limits(env_file, recorder):
    assert [len(b) for b in _batches(["x" * 10] * 150, 64)] == [64, 64, 22]
    assert [len(b) for b in _batches(["y" * 8_000] * 60, 64)] == [50, 10]  # 480,000 chars
    assert _batches(["z" * 50_000], 64) == [["z" * 8_000]]

    path = env_file(
        "CHUNKGUARD_LLM_MODEL=m\nCHUNKGUARD_EMBEDDING_MODEL=e\nCHUNKGUARD_EMBEDDING_BATCH_SIZE=2\n"
    )
    chunker_from_env(path).split("d1", DOCUMENT, "de")
    assert [len(kw["input"]) for name, kw in recorder.calls if name == "embedding"] == [2, 2]


def test_core_works_without_the_litellm_extra():
    script = (
        "import sys; sys.modules['litellm'] = None\n"
        "import chunkguard\n"
        "chunkguard.PICChunker(lambda p: 's', lambda t: [[1.0]] * len(t))\n"
        "try:\n"
        "    chunkguard.chunker_from_env\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'pip install "chunkguard[litellm]"' in result.stdout


class OpenAICompatible(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: one chat and one embeddings route.

    The first `failures` requests answer 503, like a provider under load.
    """

    failures = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if OpenAICompatible.failures > 0:
            OpenAICompatible.failures -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.endswith("/chat/completions"):
            payload = {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Resort am Meer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            payload = {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector(text)}
                    for i, text in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("failures", [0, 1])
def test_real_litellm_against_openai_compatible_server(env_file, failures):
    OpenAICompatible.failures = failures
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAICompatible)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        path = env_file(
            f"CHUNKGUARD_LLM_MODEL=openai/local-llm\nCHUNKGUARD_LLM_API_BASE={base}\n"
            "CHUNKGUARD_LLM_API_KEY=unused\n"
            f"CHUNKGUARD_EMBEDDING_MODEL=openai/local-embed\nCHUNKGUARD_EMBEDDING_API_BASE={base}\n"
            "CHUNKGUARD_EMBEDDING_API_KEY=unused\nCHUNKGUARD_MIN_CHUNK_CHARS=0\n"
            "CHUNKGUARD_MAX_RETRIES=1\nCHUNKGUARD_REQUEST_TIMEOUT=10\n"
        )
        chunks = chunker_from_env(path).split("d1", DOCUMENT, "de")
        async_chunks = asyncio.run(async_chunker_from_env(path).split("d1", DOCUMENT, "de"))
    finally:
        server.shutdown()
    assert [c.relevant for c in chunks] == [True, False, True]
    assert chunks[0].summary == "Resort am Meer"
    assert async_chunks == chunks


def test_new_limit_names_and_token_unit(env_file, recorder):
    path = env_file(
        "CHUNKGUARD_LLM_MODEL=m\nCHUNKGUARD_EMBEDDING_MODEL=e\n"
        "CHUNKGUARD_MAX_CHUNK_LENGTH=512\nCHUNKGUARD_MIN_CHUNK_LENGTH=32\n"
        "CHUNKGUARD_LENGTH_UNIT=tokens\n"
    )
    chunker = chunker_from_env(path)
    assert (chunker.max_length, chunker.min_length) == (512, 32)
    assert chunker.length("Das Resort liegt am Meer.") < len("Das Resort liegt am Meer.")


def test_token_length_counts_tokens():
    from chunkguard import token_length

    count = token_length("o200k_base")
    assert 0 < count("Hallig Hooge liegt im Wattenmeer.") < 12
