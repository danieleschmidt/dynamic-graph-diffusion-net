"""Localized messages for DGDN library."""

from typing import Dict, Any

# Message templates for all supported languages
MESSAGES: Dict[str, Dict[str, str]] = {
    'en': {
        # Training messages
        'training.started': 'Training started with {epochs} epochs',
        'training.epoch_progress': 'Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}',
        'training.validation': 'Validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}',
        'training.completed': 'Training completed successfully',
        'training.early_stop': 'Early stopping at epoch {epoch}',
        'training.checkpoint_saved': 'Checkpoint saved at {path}',
        
        # Model messages
        'model.loading': 'Loading model from {path}',
        'model.saving': 'Saving model to {path}',
        'model.created': 'DGDN model created with {layers} layers, {hidden_dim} hidden dimensions',
        'model.validation.success': 'Model validation passed',
        'model.validation.failed': 'Model validation failed: {error}',
        
        # Data messages
        'data.loading': 'Loading dataset from {path}',
        'data.preprocessing': 'Preprocessing temporal data...',
        'data.split': 'Dataset split: {train} training, {val} validation, {test} test samples',
        'data.validation.success': 'Data validation passed',
        'data.validation.failed': 'Data validation failed: {error}',
        
        # Performance messages
        'perf.optimization.enabled': 'Performance optimizations enabled: {optimizations}',
        'perf.cache.hit': 'Cache hit rate: {rate:.2%}',
        'perf.memory.usage': 'Memory usage: {usage:.1f} MB',
        'perf.speed_improvement': 'Speed improvement: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': 'Invalid input: {details}',
        'error.dimension_mismatch': 'Dimension mismatch: expected {expected}, got {actual}',
        'error.missing_attribute': 'Missing required attribute: {attribute}',
        'error.unsupported_operation': 'Unsupported operation: {operation}',
        'error.security.path_traversal': 'Security error: path traversal attempt blocked',
        
        # Security messages
        'security.validation.passed': 'Security validation passed',
        'security.input_sanitized': 'Input sanitized for security',
        'security.access_denied': 'Access denied for security reasons',
        
        # Success messages
        'success.operation_completed': 'Operation completed successfully',
        'success.benchmark_completed': 'Benchmark completed with {improvements} improvements',
        'success.tests_passed': 'All tests passed ({passed}/{total})',
    },
    
    'es': {
        # Training messages
        'training.started': 'Entrenamiento iniciado con {epochs} épocas',
        'training.epoch_progress': 'Época {epoch}/{total_epochs} - Pérdida: {loss:.4f}',
        'training.validation': 'Validación - Pérdida: {val_loss:.4f}, Precisión: {accuracy:.4f}',
        'training.completed': 'Entrenamiento completado exitosamente',
        'training.early_stop': 'Parada temprana en época {epoch}',
        'training.checkpoint_saved': 'Punto de control guardado en {path}',
        
        # Model messages
        'model.loading': 'Cargando modelo desde {path}',
        'model.saving': 'Guardando modelo en {path}',
        'model.created': 'Modelo DGDN creado con {layers} capas, {hidden_dim} dimensiones ocultas',
        'model.validation.success': 'Validación del modelo exitosa',
        'model.validation.failed': 'Validación del modelo falló: {error}',
        
        # Data messages
        'data.loading': 'Cargando conjunto de datos desde {path}',
        'data.preprocessing': 'Preprocesando datos temporales...',
        'data.split': 'División de datos: {train} entrenamiento, {val} validación, {test} prueba',
        'data.validation.success': 'Validación de datos exitosa',
        'data.validation.failed': 'Validación de datos falló: {error}',
        
        # Performance messages
        'perf.optimization.enabled': 'Optimizaciones de rendimiento habilitadas: {optimizations}',
        'perf.cache.hit': 'Tasa de acierto de caché: {rate:.2%}',
        'perf.memory.usage': 'Uso de memoria: {usage:.1f} MB',
        'perf.speed_improvement': 'Mejora de velocidad: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': 'Entrada inválida: {details}',
        'error.dimension_mismatch': 'Discrepancia de dimensiones: esperado {expected}, obtenido {actual}',
        'error.missing_attribute': 'Atributo requerido faltante: {attribute}',
        'error.unsupported_operation': 'Operación no soportada: {operation}',
        'error.security.path_traversal': 'Error de seguridad: intento de traversal de path bloqueado',
        
        # Security messages
        'security.validation.passed': 'Validación de seguridad exitosa',
        'security.input_sanitized': 'Entrada sanitizada por seguridad',
        'security.access_denied': 'Acceso denegado por razones de seguridad',
        
        # Success messages
        'success.operation_completed': 'Operación completada exitosamente',
        'success.benchmark_completed': 'Benchmark completado con {improvements} mejoras',
        'success.tests_passed': 'Todas las pruebas pasaron ({passed}/{total})',
    },
    
    'fr': {
        # Training messages
        'training.started': 'Entraînement commencé avec {epochs} époques',
        'training.epoch_progress': 'Époque {epoch}/{total_epochs} - Perte: {loss:.4f}',
        'training.validation': 'Validation - Perte: {val_loss:.4f}, Précision: {accuracy:.4f}',
        'training.completed': 'Entraînement terminé avec succès',
        'training.early_stop': 'Arrêt précoce à l\'époque {epoch}',
        'training.checkpoint_saved': 'Point de contrôle sauvegardé à {path}',
        
        # Model messages
        'model.loading': 'Chargement du modèle depuis {path}',
        'model.saving': 'Sauvegarde du modèle à {path}',
        'model.created': 'Modèle DGDN créé avec {layers} couches, {hidden_dim} dimensions cachées',
        'model.validation.success': 'Validation du modèle réussie',
        'model.validation.failed': 'Validation du modèle échouée: {error}',
        
        # Data messages
        'data.loading': 'Chargement du jeu de données depuis {path}',
        'data.preprocessing': 'Prétraitement des données temporelles...',
        'data.split': 'Division des données: {train} entraînement, {val} validation, {test} test',
        'data.validation.success': 'Validation des données réussie',
        'data.validation.failed': 'Validation des données échouée: {error}',
        
        # Performance messages
        'perf.optimization.enabled': 'Optimisations de performance activées: {optimizations}',
        'perf.cache.hit': 'Taux de succès du cache: {rate:.2%}',
        'perf.memory.usage': 'Utilisation mémoire: {usage:.1f} MB',
        'perf.speed_improvement': 'Amélioration de vitesse: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': 'Entrée invalide: {details}',
        'error.dimension_mismatch': 'Incompatibilité de dimensions: attendu {expected}, obtenu {actual}',
        'error.missing_attribute': 'Attribut requis manquant: {attribute}',
        'error.unsupported_operation': 'Opération non supportée: {operation}',
        'error.security.path_traversal': 'Erreur de sécurité: tentative de traversée de chemin bloquée',
        
        # Security messages
        'security.validation.passed': 'Validation de sécurité réussie',
        'security.input_sanitized': 'Entrée nettoyée pour la sécurité',
        'security.access_denied': 'Accès refusé pour des raisons de sécurité',
        
        # Success messages
        'success.operation_completed': 'Opération terminée avec succès',
        'success.benchmark_completed': 'Benchmark terminé avec {improvements} améliorations',
        'success.tests_passed': 'Tous les tests réussis ({passed}/{total})',
    },
    
    'de': {
        # Training messages
        'training.started': 'Training mit {epochs} Epochen gestartet',
        'training.epoch_progress': 'Epoche {epoch}/{total_epochs} - Verlust: {loss:.4f}',
        'training.validation': 'Validierung - Verlust: {val_loss:.4f}, Genauigkeit: {accuracy:.4f}',
        'training.completed': 'Training erfolgreich abgeschlossen',
        'training.early_stop': 'Frühzeitiger Stopp bei Epoche {epoch}',
        'training.checkpoint_saved': 'Checkpoint gespeichert unter {path}',
        
        # Model messages
        'model.loading': 'Modell wird geladen von {path}',
        'model.saving': 'Modell wird gespeichert unter {path}',
        'model.created': 'DGDN-Modell erstellt mit {layers} Schichten, {hidden_dim} versteckten Dimensionen',
        'model.validation.success': 'Modellvalidierung erfolgreich',
        'model.validation.failed': 'Modellvalidierung fehlgeschlagen: {error}',
        
        # Data messages
        'data.loading': 'Datensatz wird geladen von {path}',
        'data.preprocessing': 'Temporale Daten werden vorverarbeitet...',
        'data.split': 'Datenaufteilung: {train} Training, {val} Validierung, {test} Test',
        'data.validation.success': 'Datenvalidierung erfolgreich',
        'data.validation.failed': 'Datenvalidierung fehlgeschlagen: {error}',
        
        # Performance messages
        'perf.optimization.enabled': 'Leistungsoptimierungen aktiviert: {optimizations}',
        'perf.cache.hit': 'Cache-Trefferrate: {rate:.2%}',
        'perf.memory.usage': 'Speicherverbrauch: {usage:.1f} MB',
        'perf.speed_improvement': 'Geschwindigkeitsverbesserung: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': 'Ungültige Eingabe: {details}',
        'error.dimension_mismatch': 'Dimensionsfehler: erwartet {expected}, erhalten {actual}',
        'error.missing_attribute': 'Fehlendes erforderliches Attribut: {attribute}',
        'error.unsupported_operation': 'Nicht unterstützte Operation: {operation}',
        'error.security.path_traversal': 'Sicherheitsfehler: Path-Traversal-Versuch blockiert',
        
        # Security messages
        'security.validation.passed': 'Sicherheitsvalidierung erfolgreich',
        'security.input_sanitized': 'Eingabe aus Sicherheitsgründen bereinigt',
        'security.access_denied': 'Zugang aus Sicherheitsgründen verweigert',
        
        # Success messages
        'success.operation_completed': 'Operation erfolgreich abgeschlossen',
        'success.benchmark_completed': 'Benchmark abgeschlossen mit {improvements} Verbesserungen',
        'success.tests_passed': 'Alle Tests bestanden ({passed}/{total})',
    },
    
    'ja': {
        # Training messages
        'training.started': '{epochs}エポックでトレーニングを開始しました',
        'training.epoch_progress': 'エポック {epoch}/{total_epochs} - 損失: {loss:.4f}',
        'training.validation': '検証 - 損失: {val_loss:.4f}, 精度: {accuracy:.4f}',
        'training.completed': 'トレーニングが正常に完了しました',
        'training.early_stop': 'エポック{epoch}で早期停止',
        'training.checkpoint_saved': 'チェックポイントを{path}に保存しました',
        
        # Model messages
        'model.loading': '{path}からモデルを読み込み中',
        'model.saving': '{path}にモデルを保存中',
        'model.created': 'DGDNモデルを作成しました（{layers}層、{hidden_dim}隠れ次元）',
        'model.validation.success': 'モデル検証が成功しました',
        'model.validation.failed': 'モデル検証が失敗しました: {error}',
        
        # Data messages
        'data.loading': '{path}からデータセットを読み込み中',
        'data.preprocessing': '時系列データを前処理中...',
        'data.split': 'データ分割: {train}トレーニング、{val}検証、{test}テスト',
        'data.validation.success': 'データ検証が成功しました',
        'data.validation.failed': 'データ検証が失敗しました: {error}',
        
        # Performance messages
        'perf.optimization.enabled': 'パフォーマンス最適化が有効: {optimizations}',
        'perf.cache.hit': 'キャッシュヒット率: {rate:.2%}',
        'perf.memory.usage': 'メモリ使用量: {usage:.1f} MB',
        'perf.speed_improvement': '速度改善: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': '無効な入力: {details}',
        'error.dimension_mismatch': '次元の不一致: 期待値{expected}、実際{actual}',
        'error.missing_attribute': '必須属性が不足: {attribute}',
        'error.unsupported_operation': 'サポートされていない操作: {operation}',
        'error.security.path_traversal': 'セキュリティエラー: パストラバーサル攻撃をブロックしました',
        
        # Security messages
        'security.validation.passed': 'セキュリティ検証が成功しました',
        'security.input_sanitized': 'セキュリティのため入力をサニタイズしました',
        'security.access_denied': 'セキュリティ上の理由でアクセスが拒否されました',
        
        # Success messages
        'success.operation_completed': '操作が正常に完了しました',
        'success.benchmark_completed': 'ベンチマークが完了しました（{improvements}の改善）',
        'success.tests_passed': 'すべてのテストが成功しました（{passed}/{total}）',
    },
    
    'zh': {
        # Training messages
        'training.started': '开始训练，共{epochs}个轮次',
        'training.epoch_progress': '轮次 {epoch}/{total_epochs} - 损失: {loss:.4f}',
        'training.validation': '验证 - 损失: {val_loss:.4f}, 准确率: {accuracy:.4f}',
        'training.completed': '训练成功完成',
        'training.early_stop': '在轮次{epoch}提前停止',
        'training.checkpoint_saved': '检查点已保存到{path}',
        
        # Model messages
        'model.loading': '从{path}加载模型',
        'model.saving': '保存模型到{path}',
        'model.created': '创建DGDN模型：{layers}层，{hidden_dim}隐藏维度',
        'model.validation.success': '模型验证成功',
        'model.validation.failed': '模型验证失败: {error}',
        
        # Data messages
        'data.loading': '从{path}加载数据集',
        'data.preprocessing': '正在预处理时序数据...',
        'data.split': '数据划分：{train}训练，{val}验证，{test}测试',
        'data.validation.success': '数据验证成功',
        'data.validation.failed': '数据验证失败: {error}',
        
        # Performance messages
        'perf.optimization.enabled': '性能优化已启用: {optimizations}',
        'perf.cache.hit': '缓存命中率: {rate:.2%}',
        'perf.memory.usage': '内存使用: {usage:.1f} MB',
        'perf.speed_improvement': '速度提升: {improvement:.1%}',
        
        # Error messages
        'error.invalid_input': '无效输入: {details}',
        'error.dimension_mismatch': '维度不匹配：期望{expected}，实际{actual}',
        'error.missing_attribute': '缺少必需属性: {attribute}',
        'error.unsupported_operation': '不支持的操作: {operation}',
        'error.security.path_traversal': '安全错误：已阻止路径遍历攻击',
        
        # Security messages
        'security.validation.passed': '安全验证通过',
        'security.input_sanitized': '为安全起见已清理输入',
        'security.access_denied': '出于安全原因拒绝访问',
        
        # Success messages
        'success.operation_completed': '操作成功完成',
        'success.benchmark_completed': '基准测试完成，{improvements}项改进',
        'success.tests_passed': '所有测试通过（{passed}/{total}）',
    }
}


class Messages:
    """Message container for localized strings."""
    
    def __init__(self, locale: str = 'en'):
        self.locale = locale
        self.messages = MESSAGES.get(locale, MESSAGES['en'])
    
    def get(self, key: str, **kwargs) -> str:
        """Get localized message with formatting."""
        template = self.messages.get(key, key)
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            # Fallback to template without formatting if kwargs don't match
            return template
    
    def has(self, key: str) -> bool:
        """Check if message key exists."""
        return key in self.messages
    
    def set_locale(self, locale: str):
        """Change the current locale."""
        self.locale = locale
        self.messages = MESSAGES.get(locale, MESSAGES['en'])