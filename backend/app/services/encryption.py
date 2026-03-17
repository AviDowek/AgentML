"""Symmetric encryption for sensitive data (API keys, etc.)."""
import base64
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_fernet: Optional[Fernet] = None


def _get_fernet() -> Optional[Fernet]:
    """Get or create a Fernet instance from the configured encryption key."""
    global _fernet
    if _fernet is not None:
        return _fernet

    settings = get_settings()
    key = settings.api_key_encryption_key
    if not key:
        return None

    try:
        # Accept raw Fernet key (base64-encoded 32 bytes)
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
        return _fernet
    except Exception:
        logger.error("Invalid API_KEY_ENCRYPTION_KEY — must be a valid Fernet key")
        return None


def encrypt(plaintext: str) -> str:
    """Encrypt a string. Returns ciphertext prefixed with 'enc:'.

    If encryption is not configured, returns the plaintext unchanged.
    """
    f = _get_fernet()
    if f is None:
        return plaintext
    token = f.encrypt(plaintext.encode())
    return "enc:" + token.decode()


def decrypt(stored: str) -> str:
    """Decrypt a stored value. Handles both encrypted ('enc:...') and legacy plaintext values."""
    if not stored.startswith("enc:"):
        # Legacy unencrypted value — return as-is
        return stored

    f = _get_fernet()
    if f is None:
        logger.error(
            "Found encrypted API key but API_KEY_ENCRYPTION_KEY is not set. "
            "Cannot decrypt."
        )
        raise ValueError("Encryption key not configured — cannot decrypt stored API key")

    try:
        token = stored[4:].encode()  # Strip 'enc:' prefix
        return f.decrypt(token).decode()
    except InvalidToken:
        logger.error("Failed to decrypt API key — wrong encryption key or corrupted data")
        raise ValueError("Failed to decrypt API key")
