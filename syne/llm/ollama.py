"""Ollama provider — local embedding via qwen3-embedding.

Connects to Ollama's REST API (default: http://localhost:11434).
Only implements embed/embed_batch — no chat support (use other providers for chat).
"""

import httpx
import logging
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.ollama")


class OllamaProvider(LLMProvider):
    """Ollama provider for local embeddings."""

    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
    ):
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supports_vision(self) -> bool:
        return False

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
    ) -> ChatResponse:
        """Not supported — Ollama provider is embedding-only."""
        raise NotImplementedError("OllamaProvider is embedding-only. Use another provider for chat.")

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate an embedding vector using Ollama's /api/embed endpoint."""
        model = model or self.embedding_model

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()

        # Ollama returns {"embeddings": [[...]], ...}
        vector = data["embeddings"][0]
        return EmbeddingResponse(
            vector=vector,
            model=model,
            dimensions=len(vector),
            input_tokens=0,  # Ollama doesn't report token usage
        )

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts.
        
        Ollama /api/embed supports batch via input=[...].
        """
        model = model or self.embedding_model

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            EmbeddingResponse(
                vector=vec,
                model=model,
                dimensions=len(vec),
                input_tokens=0,
            )
            for vec in data["embeddings"]
        ]


async def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running and reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def check_model_available(
    model: str = "qwen3-embedding:0.6b",
    base_url: str = "http://localhost:11434",
) -> bool:
    """Check if a specific model is already pulled in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            if resp.status_code != 200:
                return False
            data = resp.json()
            model_names = [m.get("name", "") for m in data.get("models", [])]
            # Check both exact match and without tag
            model_base = model.split(":")[0]
            return any(
                m == model or m.startswith(f"{model_base}:")
                for m in model_names
            )
    except Exception:
        return False
