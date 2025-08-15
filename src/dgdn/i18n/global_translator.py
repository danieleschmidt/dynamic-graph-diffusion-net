"""
Enhanced global translation and internationalization system for DGDN.
"""

import json
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import warnings
import logging

class GlobalTranslator:
    """Advanced internationalization system supporting multiple languages and regions."""
    
    # Supported languages with their codes and names
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'region': 'Global', 'rtl': False},
        'es': {'name': 'Español', 'region': 'Global', 'rtl': False},
        'fr': {'name': 'Français', 'region': 'Europe', 'rtl': False},
        'de': {'name': 'Deutsch', 'region': 'Europe', 'rtl': False},
        'ja': {'name': '日本語', 'region': 'Asia', 'rtl': False},
        'zh': {'name': '中文', 'region': 'Asia', 'rtl': False},
        'ar': {'name': 'العربية', 'region': 'MENA', 'rtl': True},
        'pt': {'name': 'Português', 'region': 'Global', 'rtl': False},
        'ru': {'name': 'Русский', 'region': 'Europe', 'rtl': False},
        'ko': {'name': '한국어', 'region': 'Asia', 'rtl': False}
    }
    
    def __init__(self, default_language: str = 'en', translations_dir: Optional[str] = None):
        self.current_language = default_language
        self.fallback_language = 'en'
        self.translations = {}
        self.translations_dir = translations_dir or self._get_default_translations_dir()
        self.logger = logging.getLogger(f'{__name__}.GlobalTranslator')
        
        # Load translations
        self._load_all_translations()
        
        # Regional formatting
        self.regional_formats = self._initialize_regional_formats()
        
        # Pluralization rules
        self.pluralization_rules = self._initialize_pluralization_rules()
    
    def _get_default_translations_dir(self) -> str:
        """Get default translations directory."""
        current_dir = Path(__file__).parent
        return str(current_dir / 'locales')
    
    def _load_all_translations(self) -> None:
        """Load all available translations."""
        if not os.path.exists(self.translations_dir):
            os.makedirs(self.translations_dir, exist_ok=True)
            self._create_default_translations()
        
        for lang_code in self.SUPPORTED_LANGUAGES:
            self._load_language(lang_code)
    
    def _load_language(self, lang_code: str) -> None:
        """Load translations for a specific language."""
        lang_file = os.path.join(self.translations_dir, f'{lang_code}.json')
        
        if os.path.exists(lang_file):
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                self.logger.debug(f"Loaded translations for {lang_code}")
            except Exception as e:
                self.logger.error(f"Failed to load translations for {lang_code}: {e}")
                self.translations[lang_code] = {}
        else:
            self.translations[lang_code] = {}
            self.logger.warning(f"No translations file found for {lang_code}")
    
    def _create_default_translations(self) -> None:
        """Create default translation files."""
        default_messages = {
            'en': {
                'model.training.started': 'Model training started',
                'model.training.completed': 'Model training completed successfully',
                'model.training.failed': 'Model training failed',
                'model.inference.started': 'Running model inference',
                'model.inference.completed': 'Model inference completed',
                'validation.error.invalid_input': 'Invalid input provided',
                'validation.error.dimension_mismatch': 'Dimension mismatch detected',
                'validation.error.memory_limit': 'Memory limit exceeded',
                'performance.optimization.enabled': 'Performance optimization enabled',
                'performance.caching.hit': 'Cache hit',
                'performance.caching.miss': 'Cache miss',
                'security.validation.passed': 'Security validation passed',
                'security.validation.failed': 'Security validation failed',
                'global.region.compliance.gdpr': 'GDPR compliance enabled',
                'global.region.compliance.ccpa': 'CCPA compliance enabled',
                'global.region.compliance.pdpa': 'PDPA compliance enabled',
                'deployment.status.ready': 'System ready for deployment',
                'deployment.status.scaling': 'Auto-scaling in progress',
                'deployment.status.healthy': 'System health check passed',
                'common.success': 'Success',
                'common.error': 'Error',
                'common.warning': 'Warning',
                'common.info': 'Information',
                'units.time.seconds': 'seconds',
                'units.time.minutes': 'minutes',
                'units.memory.mb': 'MB',
                'units.memory.gb': 'GB',
                'units.percentage': '%'
            },
            'es': {
                'model.training.started': 'Entrenamiento del modelo iniciado',
                'model.training.completed': 'Entrenamiento del modelo completado exitosamente',
                'model.training.failed': 'Entrenamiento del modelo falló',
                'model.inference.started': 'Ejecutando inferencia del modelo',
                'model.inference.completed': 'Inferencia del modelo completada',
                'validation.error.invalid_input': 'Entrada inválida proporcionada',
                'validation.error.dimension_mismatch': 'Desajuste de dimensiones detectado',
                'validation.error.memory_limit': 'Límite de memoria excedido',
                'performance.optimization.enabled': 'Optimización de rendimiento habilitada',
                'performance.caching.hit': 'Acierto de caché',
                'performance.caching.miss': 'Falla de caché',
                'security.validation.passed': 'Validación de seguridad aprobada',
                'security.validation.failed': 'Validación de seguridad falló',
                'global.region.compliance.gdpr': 'Cumplimiento GDPR habilitado',
                'global.region.compliance.ccpa': 'Cumplimiento CCPA habilitado',
                'global.region.compliance.pdpa': 'Cumplimiento PDPA habilitado',
                'deployment.status.ready': 'Sistema listo para despliegue',
                'deployment.status.scaling': 'Auto-escalado en progreso',
                'deployment.status.healthy': 'Verificación de salud del sistema aprobada',
                'common.success': 'Éxito',
                'common.error': 'Error',
                'common.warning': 'Advertencia',
                'common.info': 'Información',
                'units.time.seconds': 'segundos',
                'units.time.minutes': 'minutos',
                'units.memory.mb': 'MB',
                'units.memory.gb': 'GB',
                'units.percentage': '%'
            },
            'fr': {
                'model.training.started': 'Entraînement du modèle commencé',
                'model.training.completed': 'Entraînement du modèle terminé avec succès',
                'model.training.failed': 'Échec de l\'entraînement du modèle',
                'model.inference.started': 'Exécution de l\'inférence du modèle',
                'model.inference.completed': 'Inférence du modèle terminée',
                'validation.error.invalid_input': 'Entrée invalide fournie',
                'validation.error.dimension_mismatch': 'Incompatibilité de dimensions détectée',
                'validation.error.memory_limit': 'Limite de mémoire dépassée',
                'performance.optimization.enabled': 'Optimisation des performances activée',
                'performance.caching.hit': 'Cache trouvé',
                'performance.caching.miss': 'Cache manqué',
                'security.validation.passed': 'Validation de sécurité réussie',
                'security.validation.failed': 'Validation de sécurité échouée',
                'global.region.compliance.gdpr': 'Conformité GDPR activée',
                'global.region.compliance.ccpa': 'Conformité CCPA activée',
                'global.region.compliance.pdpa': 'Conformité PDPA activée',
                'deployment.status.ready': 'Système prêt pour le déploiement',
                'deployment.status.scaling': 'Auto-mise à l\'échelle en cours',
                'deployment.status.healthy': 'Vérification de santé du système réussie',
                'common.success': 'Succès',
                'common.error': 'Erreur',
                'common.warning': 'Avertissement',
                'common.info': 'Information',
                'units.time.seconds': 'secondes',
                'units.time.minutes': 'minutes',
                'units.memory.mb': 'Mo',
                'units.memory.gb': 'Go',
                'units.percentage': '%'
            }
        }
        
        # Create translation files
        for lang_code, messages in default_messages.items():
            lang_file = os.path.join(self.translations_dir, f'{lang_code}.json')
            try:
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Created default translations for {lang_code}")
            except Exception as e:
                self.logger.error(f"Failed to create translations for {lang_code}: {e}")
    
    def _initialize_regional_formats(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regional formatting rules."""
        return {
            'en': {
                'decimal_separator': '.',
                'thousands_separator': ',',
                'date_format': 'MM/dd/yyyy',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '$',
                'currency_position': 'before'
            },
            'es': {
                'decimal_separator': ',',
                'thousands_separator': '.',
                'date_format': 'dd/MM/yyyy',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '€',
                'currency_position': 'after'
            },
            'fr': {
                'decimal_separator': ',',
                'thousands_separator': ' ',
                'date_format': 'dd/MM/yyyy',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '€',
                'currency_position': 'after'
            },
            'de': {
                'decimal_separator': ',',
                'thousands_separator': '.',
                'date_format': 'dd.MM.yyyy',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '€',
                'currency_position': 'after'
            },
            'ja': {
                'decimal_separator': '.',
                'thousands_separator': ',',
                'date_format': 'yyyy/MM/dd',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '¥',
                'currency_position': 'before'
            },
            'zh': {
                'decimal_separator': '.',
                'thousands_separator': ',',
                'date_format': 'yyyy/MM/dd',
                'time_format': 'HH:mm:ss',
                'currency_symbol': '¥',
                'currency_position': 'before'
            }
        }
    
    def _initialize_pluralization_rules(self) -> Dict[str, Callable]:
        """Initialize pluralization rules for different languages."""
        def english_plural(n: int) -> str:
            return 'one' if n == 1 else 'other'
        
        def spanish_plural(n: int) -> str:
            return 'one' if n == 1 else 'other'
        
        def french_plural(n: int) -> str:
            return 'one' if n in [0, 1] else 'other'
        
        def german_plural(n: int) -> str:
            return 'one' if n == 1 else 'other'
        
        def japanese_plural(n: int) -> str:
            return 'other'  # Japanese doesn't have plural forms
        
        def chinese_plural(n: int) -> str:
            return 'other'  # Chinese doesn't have plural forms
        
        return {
            'en': english_plural,
            'es': spanish_plural,
            'fr': french_plural,
            'de': german_plural,
            'ja': japanese_plural,
            'zh': chinese_plural
        }
    
    def set_language(self, lang_code: str) -> bool:
        """Set the current language."""
        if lang_code in self.SUPPORTED_LANGUAGES:
            self.current_language = lang_code
            self.logger.info(f"Language set to {lang_code}")
            return True
        else:
            self.logger.warning(f"Unsupported language: {lang_code}")
            return False
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def get_language_info(self, lang_code: Optional[str] = None) -> Dict[str, Any]:
        """Get language information."""
        lang = lang_code or self.current_language
        return self.SUPPORTED_LANGUAGES.get(lang, {})
    
    def translate(self, key: str, lang_code: Optional[str] = None, **kwargs) -> str:
        """
        Translate a message key to the specified or current language.
        
        Args:
            key: Translation key (e.g., 'model.training.started')
            lang_code: Target language code (uses current if None)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated message
        """
        target_lang = lang_code or self.current_language
        
        # Try target language first
        if target_lang in self.translations:
            message = self.translations[target_lang].get(key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Fallback to English
        if self.fallback_language in self.translations:
            message = self.translations[self.fallback_language].get(key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Last resort: return the key itself
        self.logger.warning(f"Translation not found for key: {key}")
        return key
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with variables."""
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Message formatting error: {e}")
            return message
    
    def translate_plural(self, key: str, count: int, lang_code: Optional[str] = None, **kwargs) -> str:
        """
        Translate a plural message.
        
        Args:
            key: Base translation key
            count: Number for pluralization
            lang_code: Target language code
            **kwargs: Variables for string formatting
            
        Returns:
            Translated plural message
        """
        target_lang = lang_code or self.current_language
        
        # Get pluralization rule
        plural_rule = self.pluralization_rules.get(target_lang, self.pluralization_rules['en'])
        plural_form = plural_rule(count)
        
        # Try specific plural key first
        plural_key = f"{key}.{plural_form}"
        message = self.translate(plural_key, lang_code)
        
        # If specific plural key not found, try base key
        if message == plural_key:
            message = self.translate(key, lang_code)
        
        return self._format_message(message, count=count, **kwargs)
    
    def format_number(self, number: float, lang_code: Optional[str] = None) -> str:
        """Format number according to language conventions."""
        target_lang = lang_code or self.current_language
        formats = self.regional_formats.get(target_lang, self.regional_formats['en'])
        
        # Simple number formatting (can be enhanced with locale library)
        decimal_sep = formats['decimal_separator']
        thousands_sep = formats['thousands_separator']
        
        # Split into integer and decimal parts
        parts = str(number).split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ''
        
        # Add thousands separators
        if len(integer_part) > 3:
            formatted_integer = ''
            for i, digit in enumerate(integer_part[::-1]):
                if i > 0 and i % 3 == 0:
                    formatted_integer = thousands_sep + formatted_integer
                formatted_integer = digit + formatted_integer
        else:
            formatted_integer = integer_part
        
        # Combine parts
        if decimal_part:
            return f"{formatted_integer}{decimal_sep}{decimal_part}"
        else:
            return formatted_integer
    
    def format_currency(self, amount: float, lang_code: Optional[str] = None) -> str:
        """Format currency according to language conventions."""
        target_lang = lang_code or self.current_language
        formats = self.regional_formats.get(target_lang, self.regional_formats['en'])
        
        formatted_number = self.format_number(amount, lang_code)
        currency_symbol = formats['currency_symbol']
        currency_position = formats['currency_position']
        
        if currency_position == 'before':
            return f"{currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {currency_symbol}"
    
    def get_available_languages(self) -> List[Dict[str, str]]:
        """Get list of available languages."""
        return [
            {
                'code': code,
                'name': info['name'],
                'region': info['region'],
                'rtl': info['rtl']
            }
            for code, info in self.SUPPORTED_LANGUAGES.items()
        ]
    
    def get_region_languages(self, region: str) -> List[str]:
        """Get languages for a specific region."""
        return [
            code for code, info in self.SUPPORTED_LANGUAGES.items()
            if info['region'] == region or info['region'] == 'Global'
        ]
    
    def validate_translation_completeness(self) -> Dict[str, Any]:
        """Validate translation completeness across all languages."""
        if 'en' not in self.translations:
            return {'error': 'English translations not available for validation'}
        
        english_keys = set(self.translations['en'].keys())
        completeness_report = {}
        
        for lang_code in self.SUPPORTED_LANGUAGES:
            if lang_code == 'en':
                completeness_report[lang_code] = {'completeness': 100.0, 'missing_keys': []}
                continue
            
            lang_keys = set(self.translations.get(lang_code, {}).keys())
            missing_keys = english_keys - lang_keys
            completeness = ((len(english_keys) - len(missing_keys)) / len(english_keys)) * 100
            
            completeness_report[lang_code] = {
                'completeness': completeness,
                'missing_keys': list(missing_keys)
            }
        
        return completeness_report
    
    def auto_detect_language(self, text: str) -> str:
        """Simple language detection (placeholder for real implementation)."""
        # This is a simple heuristic - in production, use a proper language detection library
        
        # Check for common patterns
        if any(char in text for char in '你好我是中文'):
            return 'zh'
        elif any(char in text for char in 'こんにちは日本語'):
            return 'ja'
        elif any(char in text for char in 'أهلاًوسهلاًالعربية'):
            return 'ar'
        elif 'ñ' in text or 'á' in text or 'é' in text:
            return 'es'
        elif 'ç' in text or 'à' in text or 'è' in text:
            return 'fr'
        elif 'ü' in text or 'ä' in text or 'ö' in text:
            return 'de'
        else:
            return 'en'  # Default to English

# Global translator instance
_global_translator = None

def get_global_translator() -> GlobalTranslator:
    """Get the global translator instance."""
    global _global_translator
    if _global_translator is None:
        _global_translator = GlobalTranslator()
    return _global_translator

def set_global_language(lang_code: str) -> bool:
    """Set the global language."""
    return get_global_translator().set_language(lang_code)

def translate(key: str, lang_code: Optional[str] = None, **kwargs) -> str:
    """Global translate function."""
    return get_global_translator().translate(key, lang_code, **kwargs)

def translate_plural(key: str, count: int, lang_code: Optional[str] = None, **kwargs) -> str:
    """Global plural translate function."""
    return get_global_translator().translate_plural(key, count, lang_code, **kwargs)

def format_number(number: float, lang_code: Optional[str] = None) -> str:
    """Global number formatting function."""
    return get_global_translator().format_number(number, lang_code)

def format_currency(amount: float, lang_code: Optional[str] = None) -> str:
    """Global currency formatting function."""
    return get_global_translator().format_currency(amount, lang_code)