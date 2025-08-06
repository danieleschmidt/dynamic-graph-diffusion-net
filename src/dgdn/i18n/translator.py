"""Translation system for DGDN library."""

import os
import threading
from typing import Optional, Dict, Any
from .messages import Messages, MESSAGES
from .locales import SUPPORTED_LOCALES, get_default_locale, is_supported_locale

# Global translator instance
_translator_instance: Optional['DGDNTranslator'] = None
_translator_lock = threading.Lock()


class DGDNTranslator:
    """Main translation system for DGDN library."""
    
    def __init__(self, locale: Optional[str] = None):
        """Initialize translator with locale detection."""
        self._locale = self._detect_locale(locale)
        self._messages = Messages(self._locale)
        self._fallback_locale = get_default_locale()
        self._fallback_messages = Messages(self._fallback_locale)
    
    def _detect_locale(self, provided_locale: Optional[str]) -> str:
        """Detect locale from various sources."""
        # Priority order:
        # 1. Explicitly provided locale
        # 2. Environment variable DGDN_LOCALE
        # 3. Environment variable LANG
        # 4. System locale detection
        # 5. Default fallback
        
        if provided_locale and is_supported_locale(provided_locale):
            return provided_locale
        
        # Check environment variables
        env_locale = os.environ.get('DGDN_LOCALE')
        if env_locale and is_supported_locale(env_locale):
            return env_locale
        
        # Check system LANG variable
        lang_var = os.environ.get('LANG', '')
        if lang_var:
            # Extract language code (e.g., 'en_US.UTF-8' -> 'en')
            lang_code = lang_var.split('_')[0].split('.')[0]
            if is_supported_locale(lang_code):
                return lang_code
        
        # Try to detect system locale
        try:
            import locale
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                lang_code = system_locale.split('_')[0]
                if is_supported_locale(lang_code):
                    return lang_code
        except (ImportError, TypeError):
            pass
        
        # Fallback to default
        return get_default_locale()
    
    @property
    def locale(self) -> str:
        """Get current locale."""
        return self._locale
    
    @locale.setter
    def locale(self, value: str):
        """Set new locale."""
        if is_supported_locale(value):
            self._locale = value
            self._messages = Messages(value)
        else:
            raise ValueError(f"Unsupported locale: {value}. Supported: {SUPPORTED_LOCALES}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate message key with formatting."""
        # Try current locale first
        if self._messages.has(key):
            return self._messages.get(key, **kwargs)
        
        # Fallback to default locale
        if self._locale != self._fallback_locale:
            if self._fallback_messages.has(key):
                return self._fallback_messages.get(key, **kwargs)
        
        # Ultimate fallback: return key itself
        return key
    
    def t(self, key: str, **kwargs) -> str:
        """Short alias for translate()."""
        return self.translate(key, **kwargs)
    
    def has_translation(self, key: str) -> bool:
        """Check if translation exists for key."""
        return self._messages.has(key) or self._fallback_messages.has(key)
    
    def get_supported_locales(self) -> set:
        """Get list of supported locales."""
        return SUPPORTED_LOCALES.copy()
    
    def get_locale_info(self, locale: Optional[str] = None) -> Dict[str, str]:
        """Get information about current or specified locale."""
        from .locales import get_locale_info
        target_locale = locale or self._locale
        return get_locale_info(target_locale)
    
    def format_number(self, number: float, decimal_places: int = 2) -> str:
        """Format number according to locale conventions."""
        # Simplified number formatting - in production, use proper locale formatting
        if self._locale in ['de']:
            # German uses comma as decimal separator
            formatted = f"{number:.{decimal_places}f}".replace('.', ',')
        elif self._locale in ['fr']:
            # French uses comma as decimal separator and space as thousands separator
            formatted = f"{number:,.{decimal_places}f}".replace(',', ' ').replace('.', ',')
        else:
            # Default English formatting
            formatted = f"{number:,.{decimal_places}f}"
        
        return formatted
    
    def format_percentage(self, value: float, decimal_places: int = 1) -> str:
        """Format percentage according to locale."""
        percentage = value * 100
        formatted_num = self.format_number(percentage, decimal_places)
        
        if self._locale == 'fr':
            return f"{formatted_num} %"  # French has space before %
        else:
            return f"{formatted_num}%"
    
    def pluralize(self, key: str, count: int, **kwargs) -> str:
        """Handle pluralization (simplified)."""
        # This is a simplified implementation
        # In production, use proper pluralization rules for each language
        plural_key = f"{key}.plural" if count != 1 else key
        
        if self.has_translation(plural_key):
            return self.translate(plural_key, count=count, **kwargs)
        else:
            return self.translate(key, count=count, **kwargs)


def get_translator(locale: Optional[str] = None) -> DGDNTranslator:
    """Get global translator instance (singleton pattern)."""
    global _translator_instance
    
    with _translator_lock:
        if _translator_instance is None or (locale and _translator_instance.locale != locale):
            _translator_instance = DGDNTranslator(locale)
        return _translator_instance


def set_global_locale(locale: str):
    """Set global locale for all translations."""
    global _translator_instance
    
    with _translator_lock:
        if _translator_instance is None:
            _translator_instance = DGDNTranslator(locale)
        else:
            _translator_instance.locale = locale


# Convenience functions for common operations
def _(key: str, **kwargs) -> str:
    """Global translation function (gettext-style)."""
    return get_translator().translate(key, **kwargs)


def ngettext(singular_key: str, plural_key: str, n: int, **kwargs) -> str:
    """Ngettext-style pluralization."""
    translator = get_translator()
    key = plural_key if n != 1 else singular_key
    return translator.translate(key, n=n, **kwargs)


def format_metric(metric_name: str, value: float, locale: Optional[str] = None) -> str:
    """Format metric values with proper localization."""
    translator = get_translator(locale)
    
    # Common metric formatting
    if 'percentage' in metric_name or 'rate' in metric_name:
        return translator.format_percentage(value)
    elif 'memory' in metric_name or 'size' in metric_name:
        # Handle memory/size formatting
        if value >= 1024:
            return f"{translator.format_number(value / 1024, 1)} GB"
        else:
            return f"{translator.format_number(value, 1)} MB"
    elif 'time' in metric_name:
        # Handle time formatting
        if value < 1:
            return f"{translator.format_number(value * 1000, 0)} ms"
        else:
            return f"{translator.format_number(value, 2)} s"
    else:
        return translator.format_number(value)


# Example usage helper
def demonstrate_i18n():
    """Demonstrate internationalization features."""
    print("🌍 DGDN Internationalization Demo")
    print("=" * 50)
    
    for locale in SUPPORTED_LOCALES:
        translator = DGDNTranslator(locale)
        print(f"\n📍 {locale.upper()} ({translator.get_locale_info()['native_name']}):")
        print(f"  Training: {translator.t('training.started', epochs=10)}")
        print(f"  Model: {translator.t('model.created', layers=3, hidden_dim=256)}")
        print(f"  Performance: {translator.t('perf.speed_improvement', improvement=0.27)}")
        print(f"  Success: {translator.t('success.tests_passed', passed=25, total=30)}")


if __name__ == "__main__":
    demonstrate_i18n()