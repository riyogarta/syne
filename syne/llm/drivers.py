"""Driver registry for LLM providers.

Maps driver names to provider classes and handles provider instantiation
from model registry entries.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Type

from .provider import LLMProvider, ChatMessage, ChatResponse
from .google import GoogleProvider
from .codex import CodexProvider
from .openai import OpenAIProvider
from .together import TogetherProvider
from .hybrid import HybridProvider
from .anthropic import AnthropicProvider

logger = logging.getLogger("syne.llm.drivers")

# Driver name → Provider class mapping
_DRIVER_MAP: dict[str, Type[LLMProvider]] = {
    "google_cca": GoogleProvider,
    "codex": CodexProvider,
    "openai_compat": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_driver(driver_name: str) -> Optional[Type[LLMProvider]]:
    """Get the provider class for a driver name.
    
    Args:
        driver_name: Driver identifier (e.g., 'google_cca', 'codex', 'openai_compat')
        
    Returns:
        Provider class or None if not found
    """
    return _DRIVER_MAP.get(driver_name)


def list_drivers() -> list[str]:
    """List all available driver names."""
    return list(_DRIVER_MAP.keys())


async def create_provider(model_entry: dict) -> LLMProvider:
    """Instantiate a provider from a model registry entry.
    
    Args:
        model_entry: Dict with keys:
            - driver: Driver name ('google_cca', 'codex', 'openai_compat')
            - model_id: Model identifier for the provider
            - auth: Auth method ('oauth', 'api_key')
            - base_url: (optional) API base URL for openai_compat
            - credential_key: (optional) Config key for API credentials
            
    Returns:
        Configured LLMProvider instance
        
    Raises:
        RuntimeError: If driver not found or configuration error
    """
    driver_name = model_entry.get("driver")
    model_id = model_entry.get("model_id")
    
    if not driver_name:
        raise RuntimeError(f"Model entry missing 'driver': {model_entry}")
    
    driver_class = get_driver(driver_name)
    if not driver_class:
        raise RuntimeError(f"Unknown driver: {driver_name}")
    
    # ═══════════════════════════════════════════════════════════════
    # Google CCA (OAuth)
    # ═══════════════════════════════════════════════════════════════
    if driver_name == "google_cca":
        from ..auth.google_oauth import get_credentials
        creds = await get_credentials()
        return GoogleProvider(
            credentials=creds,
            chat_model=model_id,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Codex (ChatGPT OAuth)
    # ═══════════════════════════════════════════════════════════════
    elif driver_name == "codex":
        access_token, refresh_token = _load_codex_tokens()
        return CodexProvider(
            access_token=access_token,
            refresh_token=refresh_token,
            chat_model=model_id,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Anthropic Claude (OAuth via claude.ai)
    # ═══════════════════════════════════════════════════════════════
    elif driver_name == "anthropic":
        return AnthropicProvider(
            chat_model=model_id,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # OpenAI-Compatible (Groq, Together, OpenRouter, etc.)
    # ═══════════════════════════════════════════════════════════════
    elif driver_name == "openai_compat":
        base_url = model_entry.get("base_url", "https://api.openai.com/v1")
        credential_key = model_entry.get("credential_key")
        
        # Load API key from config or environment
        api_key = await _load_api_key(credential_key)
        if not api_key:
            raise RuntimeError(
                f"No API key found for {model_entry.get('label', model_id)}. "
                f"Set {credential_key} in config or corresponding env var."
            )
        
        # Determine provider name from base_url
        provider_name = "openai"
        if "groq.com" in base_url:
            provider_name = "groq"
        elif "together" in base_url:
            provider_name = "together"
        elif "openrouter" in base_url:
            provider_name = "openrouter"
        
        return OpenAIProvider(
            api_key=api_key,
            chat_model=model_id,
            base_url=base_url,
            provider_name=provider_name,
        )
    
    raise RuntimeError(f"Unhandled driver: {driver_name}")


async def create_embedding_provider(embed_entry: dict) -> Optional[LLMProvider]:
    """Instantiate an embedding provider from an embedding registry entry.
    
    Args:
        embed_entry: Dict with keys:
            - driver: 'together' or 'openai_compat'
            - model_id: Embedding model name
            - auth: 'api_key'
            - credential_key: Config key for API credentials
            - base_url: (optional) API base URL for openai_compat
            - dimensions: (optional) Embedding dimensions
            
    Returns:
        Configured LLMProvider instance or None on failure
    """
    driver_name = embed_entry.get("driver")
    model_id = embed_entry.get("model_id")
    credential_key = embed_entry.get("credential_key")
    
    if not driver_name or not model_id:
        logger.error(f"Embedding entry missing driver/model_id: {embed_entry}")
        return None
    
    api_key = await _load_api_key(credential_key)
    if not api_key:
        logger.error(f"No API key for embedding provider {embed_entry.get('label', model_id)}")
        return None
    
    if driver_name == "together":
        return TogetherProvider(
            api_key=api_key,
            embedding_model=model_id,
        )
    elif driver_name == "openai_compat":
        base_url = embed_entry.get("base_url", "https://api.openai.com/v1")
        return OpenAIProvider(
            api_key=api_key,
            chat_model="unused",
            base_url=base_url,
            provider_name="embedding",
        )
    else:
        logger.error(f"Unsupported embedding driver: {driver_name}")
        return None


async def create_hybrid_provider(model_entry: dict) -> LLMProvider:
    """Create a HybridProvider with chat from model_entry and embedding from registry.
    
    Uses the embedding model registry (provider.embedding_models + provider.active_embedding)
    to select the embedding provider. Falls back to Together AI if registry not set.
    
    Args:
        model_entry: Model registry entry for chat provider
        
    Returns:
        HybridProvider with chat and embedding capabilities
    """
    from ..db.models import get_config
    
    chat_provider = await create_provider(model_entry)
    
    # ═══════════════════════════════════════════════════════════════
    # Try embedding model registry first
    # ═══════════════════════════════════════════════════════════════
    embed_models = await get_config("provider.embedding_models", None)
    active_embed_key = await get_config("provider.active_embedding", None)
    
    if embed_models and active_embed_key:
        embed_entry = get_model_from_list(embed_models, active_embed_key)
        if embed_entry:
            embed_provider = await create_embedding_provider(embed_entry)
            if embed_provider:
                logger.info(f"Using embedding: {embed_entry.get('label', active_embed_key)}")
                return HybridProvider(chat_provider=chat_provider, embed_provider=embed_provider)
            else:
                logger.warning(f"Failed to create embedding provider '{active_embed_key}', falling back")
    
    # ═══════════════════════════════════════════════════════════════
    # Fallback: Together AI (legacy behavior)
    # ═══════════════════════════════════════════════════════════════
    together_key = os.environ.get("TOGETHER_API_KEY")
    if not together_key:
        try:
            oc_config = json.loads(
                open(os.path.expanduser("~/.openclaw/openclaw.json")).read()
            )
            together_key = oc_config.get("env", {}).get("TOGETHER_API_KEY")
        except Exception:
            pass
    
    if together_key:
        embed_provider = TogetherProvider(
            api_key=together_key,
            embedding_model="BAAI/bge-base-en-v1.5",
        )
        return HybridProvider(chat_provider=chat_provider, embed_provider=embed_provider)
    else:
        logger.warning(
            "No embedding provider available. Memory search will not work. "
            "Set TOGETHER_API_KEY for embeddings."
        )
        return chat_provider


def _load_codex_tokens() -> tuple[str, str]:
    """Load Codex OAuth tokens from OpenClaw auth store.
    
    Returns:
        Tuple of (access_token, refresh_token)
        
    Raises:
        RuntimeError: If tokens not found
    """
    auth_path = os.path.expanduser("~/.openclaw/agents/main/agent/auth.json")
    try:
        with open(auth_path) as f:
            auth_data = json.load(f)
        codex_auth = auth_data.get("openai-codex", {})
        access_token = codex_auth.get("access", "")
        refresh_token = codex_auth.get("refresh", "")
        if not access_token:
            raise RuntimeError("No Codex OAuth token found in OpenClaw auth store.")
        return access_token, refresh_token
    except FileNotFoundError:
        raise RuntimeError(
            f"OpenClaw auth file not found: {auth_path}. "
            "Run OpenClaw with openai-codex OAuth first."
        )


async def _load_api_key(credential_key: Optional[str]) -> Optional[str]:
    """Load API key from config or environment.
    
    Args:
        credential_key: Config key like 'credential.groq_api_key'
        
    Returns:
        API key string or None
    """
    from ..db.models import get_config
    
    # Try DB config first
    if credential_key:
        key = await get_config(credential_key)
        if key:
            return key
    
    # Try environment variables
    if credential_key:
        # Convert 'credential.groq_api_key' → 'GROQ_API_KEY'
        env_name = credential_key.split(".")[-1].upper()
        env_val = os.environ.get(env_name)
        if env_val:
            return env_val
    
    # Try OpenClaw config
    try:
        oc_config = json.loads(
            open(os.path.expanduser("~/.openclaw/openclaw.json")).read()
        )
        if credential_key:
            env_name = credential_key.split(".")[-1].upper()
            key = oc_config.get("env", {}).get(env_name)
            if key:
                return key
    except Exception:
        pass
    
    return None


async def test_embedding(embed_entry: dict, timeout: int = 10) -> tuple[bool, str]:
    """Test an embedding provider by generating a small embedding.
    
    Args:
        embed_entry: Embedding registry entry dict
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        provider = await create_embedding_provider(embed_entry)
        if not provider:
            return False, "Failed to create provider (missing API key?)"
        
        model_id = embed_entry.get("model_id", "")
        result = await asyncio.wait_for(
            provider.embed("test embedding", model=model_id),
            timeout=timeout,
        )
        
        if result and result.embedding and len(result.embedding) > 0:
            logger.info(f"Embedding test passed: {embed_entry.get('label')} → {len(result.embedding)} dims")
            return True, ""
        else:
            return False, "Empty embedding response"
            
    except asyncio.TimeoutError:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            error_msg = f"HTTP {e.response.status_code}: {error_msg[:100]}"
        return False, error_msg[:200]


async def test_model(provider: LLMProvider, timeout: int = 10) -> tuple[bool, str]:
    """Send a simple test message to verify provider works.
    
    Args:
        provider: LLMProvider instance to test
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, error_message)
        - success=True, error_message="" on success
        - success=False, error_message="..." on failure
    """
    test_messages = [
        ChatMessage(role="user", content="Compatibility test. Reply OK.")
    ]
    
    try:
        response = await asyncio.wait_for(
            provider.chat(test_messages, max_tokens=10),
            timeout=timeout,
        )
        
        if response and response.content and len(response.content.strip()) > 0:
            logger.info(f"Model test passed: {provider.name} → '{response.content[:50]}'")
            return True, ""
        else:
            return False, "Empty response from model"
            
    except asyncio.TimeoutError:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        error_msg = str(e)
        # Extract useful part from HTTP errors
        if "HTTPStatusError" in type(e).__name__ or "status" in error_msg.lower():
            # Try to get status code
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                error_msg = f"HTTP {e.response.status_code}: {error_msg[:100]}"
        return False, error_msg[:200]


def get_model_from_list(models: list[dict], model_key: str) -> Optional[dict]:
    """Find a model entry by its key.
    
    Args:
        models: List of model registry entries
        model_key: The 'key' field to search for
        
    Returns:
        Model entry dict or None
    """
    for model in models:
        if model.get("key") == model_key:
            return model
    return None
