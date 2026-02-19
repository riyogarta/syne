"""Voice Tool — speech-to-text transcription and text-to-speech."""

import logging
import httpx
import os
from pathlib import Path
from typing import Optional

from ..db.models import get_config

logger = logging.getLogger("syne.tools.voice")

# Default STT settings
DEFAULT_STT_PROVIDER = "groq"
DEFAULT_STT_MODEL = "whisper-large-v3"
GROQ_WHISPER_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"


async def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from config, with fallbacks.
    
    Priority:
    1. credential.groq_api_key (DB)
    2. GROQ_API_KEY environment variable
    3. credential.openai_compat_api_key (fallback if Groq was chosen during init)
    """
    # Check DB first
    api_key = await get_config("credential.groq_api_key", None)
    if api_key:
        return api_key
    
    # Check environment
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    # Fallback to openai_compat if user chose Groq during init
    api_key = await get_config("credential.openai_compat_api_key", None)
    if api_key:
        return api_key
    
    return None


async def transcribe_audio(
    audio_data: bytes,
    filename: str = "audio.ogg",
    language: str = "",
) -> tuple[bool, str]:
    """Transcribe audio using configured STT provider.
    
    Args:
        audio_data: Raw audio bytes (OGG, MP3, WAV, etc.)
        filename: Original filename for format detection
        language: Optional language hint (ISO 639-1 code, e.g. "id" for Indonesian)
        
    Returns:
        Tuple of (success, transcription_or_error)
    """
    # Get STT config
    stt_provider = await get_config("voice.stt_provider", DEFAULT_STT_PROVIDER)
    stt_model = await get_config("voice.stt_model", DEFAULT_STT_MODEL)
    
    if stt_provider == "groq":
        return await _transcribe_groq(audio_data, filename, stt_model, language)
    else:
        return False, f"Unknown STT provider: {stt_provider}"


async def _transcribe_groq(
    audio_data: bytes,
    filename: str,
    model: str,
    language: str,
) -> tuple[bool, str]:
    """Transcribe audio using Groq Whisper API.
    
    Args:
        audio_data: Raw audio bytes
        filename: Original filename
        model: Whisper model to use
        language: Optional language hint
        
    Returns:
        Tuple of (success, transcription_or_error)
    """
    api_key = await get_groq_api_key()
    
    if not api_key:
        return False, (
            "Groq API key not configured. Set credential.groq_api_key via: "
            "update_config(action='set', key='credential.groq_api_key', value='YOUR_KEY')"
        )
    
    # Prepare multipart form data
    # Groq Whisper API follows OpenAI's format
    files = {
        "file": (filename, audio_data, "audio/ogg"),
    }
    data = {
        "model": model,
    }
    
    if language:
        data["language"] = language
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                GROQ_WHISPER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                files=files,
                data=data,
            )
            
            if response.status_code != 200:
                error_text = response.text[:300]
                logger.error(f"Groq Whisper API error: {response.status_code} - {error_text}")
                return False, f"Transcription failed: {error_text}"
            
            result = response.json()
            text = result.get("text", "").strip()
            
            if not text:
                return False, "Transcription returned empty text"
            
            logger.info(f"Transcribed {len(audio_data)} bytes -> {len(text)} chars")
            return True, text
            
    except httpx.TimeoutException:
        return False, "Transcription request timed out (60s)"
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return False, f"Transcription error: {str(e)}"


async def send_voice_handler(
    text: str,
    chat_id: str = "",
) -> str:
    """Send a voice message (text-to-speech).
    
    Args:
        text: Text to convert to speech
        chat_id: Target chat ID
        
    Returns:
        Success or error message
    """
    # TODO: Implement TTS sending
    # Options:
    # 1. Google Gemini TTS (if available via OAuth)
    # 2. ElevenLabs (requires API key)
    # 3. OpenAI TTS (requires API key)
    
    return (
        "TTS not yet configured. Voice message sending is planned for a future update. "
        "For now, I can only receive and transcribe voice messages."
    )


# Tool metadata for registration
SEND_VOICE_TOOL = {
    "name": "send_voice",
    "description": "Send a voice message (text-to-speech). Currently not implemented — voice receiving works, but TTS sending is TODO.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to convert to speech and send as voice message",
            },
            "chat_id": {
                "type": "string",
                "description": "Target chat ID",
            },
        },
        "required": ["text"],
    },
    "handler": send_voice_handler,
    "requires_access_level": "owner",  # Rule 700 — voice represents the bot
}
