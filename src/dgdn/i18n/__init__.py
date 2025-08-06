"""Internationalization (I18n) support for DGDN."""

from .translator import DGDNTranslator, get_translator, set_global_locale
from .messages import Messages
from .locales import SUPPORTED_LOCALES

__all__ = ["DGDNTranslator", "get_translator", "set_global_locale", "Messages", "SUPPORTED_LOCALES"]