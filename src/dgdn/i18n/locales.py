"""Supported locales and language configurations."""

from typing import Dict, Set

# Supported locales as specified in TERRAGON SDLC
SUPPORTED_LOCALES: Set[str] = {
    'en',    # English (default)
    'es',    # Spanish
    'fr',    # French
    'de',    # German
    'ja',    # Japanese
    'zh'     # Chinese
}

# Locale metadata
LOCALE_INFO: Dict[str, Dict[str, str]] = {
    'en': {
        'name': 'English',
        'native_name': 'English',
        'direction': 'ltr',
        'region': 'US'
    },
    'es': {
        'name': 'Spanish',
        'native_name': 'Español',
        'direction': 'ltr',
        'region': 'ES'
    },
    'fr': {
        'name': 'French', 
        'native_name': 'Français',
        'direction': 'ltr',
        'region': 'FR'
    },
    'de': {
        'name': 'German',
        'native_name': 'Deutsch', 
        'direction': 'ltr',
        'region': 'DE'
    },
    'ja': {
        'name': 'Japanese',
        'native_name': '日本語',
        'direction': 'ltr',
        'region': 'JP'
    },
    'zh': {
        'name': 'Chinese',
        'native_name': '中文',
        'direction': 'ltr',
        'region': 'CN'
    }
}

def get_locale_info(locale: str) -> Dict[str, str]:
    """Get information about a specific locale."""
    return LOCALE_INFO.get(locale, LOCALE_INFO['en'])

def is_supported_locale(locale: str) -> bool:
    """Check if a locale is supported."""
    return locale in SUPPORTED_LOCALES

def get_default_locale() -> str:
    """Get the default locale."""
    return 'en'