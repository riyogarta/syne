"""Model resolution utilities for per-user and per-group model selection."""

import logging
from typing import Optional, Dict

logger = logging.getLogger("syne.llm.model_resolver")


async def get_effective_model(user_id: str, group_id: Optional[str] = None) -> str:
    """Get the effective model for a user/group context.

    Priority:
    1. User-specific model (provider.user.{user_id}.chat_model)
    2. Group-specific model (provider.group.{group_id}.chat_model)
    3. Global default (provider.chat_model)

    Args:
        user_id: User platform ID
        group_id: Group ID (optional, for group conversations)

    Returns:
        Model key (e.g. "claude-3-5-sonnet-20241022")
    """
    from ..db.models import get_config

    # Check user-specific model first
    user_model_key = f"provider.user.{user_id}.chat_model"
    user_model = await get_config(user_model_key)
    if user_model:
        logger.debug(f"Using user-specific model for {user_id}: {user_model}")
        return user_model

    # Check group-specific model if in a group
    if group_id:
        group_model_key = f"provider.group.{group_id}.chat_model"
        group_model = await get_config(group_model_key)
        if group_model:
            logger.debug(f"Using group-specific model for {group_id}: {group_model}")
            return group_model

    # Fall back to global default
    global_model = await get_config("provider.chat_model")
    if global_model:
        logger.debug(f"Using global default model: {global_model}")
        return global_model

    # Ultimate fallback
    logger.warning("No model configured, using fallback")
    return "claude-3-5-sonnet-20241022"


async def get_available_models() -> Dict[str, str]:
    """Get dict of available model keys and their descriptions.

    Returns:
        Dict mapping model keys to descriptions
    """
    return {
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Latest) - Most capable",
        "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Latest) - Fast & efficient",
        "claude-3-haiku-20240307": "Claude 3 Haiku - Ultra fast",
        "gpt-4o": "GPT-4o - OpenAI's most capable",
        "gpt-4o-mini": "GPT-4o Mini - Fast & affordable",
        "gpt-4-turbo": "GPT-4 Turbo - Previous generation",
        "gemini-1.5-pro-latest": "Gemini 1.5 Pro - Google's best",
        "gemini-1.5-flash-latest": "Gemini 1.5 Flash - Fast Google model",
        "llama-3.1-70b-versatile": "Llama 3.1 70B - Meta's large model",
        "llama-3.1-8b-instant": "Llama 3.1 8B - Fast Meta model",
        "mixtral-8x7b-32768": "Mixtral 8x7B - Mistral's MoE model",
        "qwen3:0.6b": "Qwen 3 0.6B - Local tiny model",
        "llama3.2:3b": "Llama 3.2 3B - Local small model",
        "codestral-latest": "Codestral - Code specialist",
        "command-r-plus": "Command R+ - Cohere's model",
    }


async def set_user_model(user_id: str, model: Optional[str]) -> bool:
    """Set preferred model for a user.

    Args:
        user_id: User platform ID
        model: Model key to set, or None to clear

    Returns:
        True if successful
    """
    if model is None:
        return await remove_user_model(user_id)

    from ..db.models import set_config

    available_models = await get_available_models()
    if model not in available_models:
        logger.error(f"Invalid model: {model}. Available: {list(available_models.keys())}")
        return False

    config_key = f"provider.user.{user_id}.chat_model"
    await set_config(config_key, model)
    logger.info(f"Set user {user_id} model to {model}")
    return True


async def set_group_model(group_id: str, model: str) -> bool:
    """Set preferred model for a group.

    Args:
        group_id: Group ID
        model: Model key to set

    Returns:
        True if successful
    """
    from ..db.models import set_config

    available_models = await get_available_models()
    if model not in available_models:
        logger.error(f"Invalid model: {model}. Available: {list(available_models.keys())}")
        return False

    config_key = f"provider.group.{group_id}.chat_model"
    await set_config(config_key, model)
    logger.info(f"Set group {group_id} model to {model}")
    return True


async def get_user_model(user_id: str) -> Optional[str]:
    """Get user-specific model if set.

    Args:
        user_id: User platform ID

    Returns:
        Model key if set, None otherwise
    """
    from ..db.models import get_config

    config_key = f"provider.user.{user_id}.chat_model"
    return await get_config(config_key)


async def get_group_model(group_id: str) -> Optional[str]:
    """Get group-specific model if set.

    Args:
        group_id: Group ID

    Returns:
        Model key if set, None otherwise
    """
    from ..db.models import get_config

    config_key = f"provider.group.{group_id}.chat_model"
    return await get_config(config_key)


async def remove_user_model(user_id: str) -> bool:
    """Remove user-specific model preference.

    Args:
        user_id: User platform ID

    Returns:
        True if successful
    """
    from ..db.models import delete_config

    config_key = f"provider.user.{user_id}.chat_model"
    await delete_config(config_key)
    logger.info(f"Removed user {user_id} model preference")
    return True


async def remove_group_model(group_id: str) -> bool:
    """Remove group-specific model preference.

    Args:
        group_id: Group ID

    Returns:
        True if successful
    """
    from ..db.models import delete_config

    config_key = f"provider.group.{group_id}.chat_model"
    await delete_config(config_key)
    logger.info(f"Removed group {group_id} model preference")
    return True


async def get_user_model_info(user_id: str) -> Dict[str, any]:
    """Get comprehensive model info for a user.

    Args:
        user_id: User platform ID

    Returns:
        Dict with model info: {
            'user_model': user-specific model key or None,
            'effective_model': the actual model that will be used,
            'display_name': human-readable model name,
            'is_custom': whether user has a custom model set
        }
    """
    available_models = await get_available_models()
    
    user_model = await get_user_model(user_id)
    effective_model = await get_effective_model(user_id)
    
    return {
        'user_model': user_model,
        'effective_model': effective_model,
        'display_name': available_models.get(effective_model, effective_model),
        'is_custom': user_model is not None
    }