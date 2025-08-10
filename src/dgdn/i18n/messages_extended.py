"""Extended multi-language message definitions for DGDN."""

# English (default)
MESSAGES_EN = {
    # Training messages
    "training.started": "Training started with {epochs} epochs",
    "training.epoch_complete": "Epoch {epoch}/{total_epochs} completed - Loss: {loss:.4f}",
    "training.early_stopping": "Early stopping triggered at epoch {epoch}",
    "training.completed": "Training completed successfully",
    "training.failed": "Training failed: {error}",
    
    # Model messages
    "model.created": "DGDN model created with {parameters:,} parameters",
    "model.loaded": "Model loaded from {path}",
    "model.saved": "Model saved to {path}",
    "model.validation_failed": "Model validation failed: {error}",
    
    # Data processing
    "data.loaded": "Dataset loaded: {nodes} nodes, {edges} edges",
    "data.preprocessing": "Preprocessing temporal graph data...",
    "data.validation_error": "Data validation error: {error}",
    
    # Performance
    "perf.inference_time": "Inference completed in {time:.3f}s",
    "perf.memory_usage": "Memory usage: {memory:.1f} MB",
    "perf.throughput": "Throughput: {rps:.1f} requests/second",
    
    # Security
    "security.access_granted": "Access granted to user {user_id}",
    "security.access_denied": "Access denied for user {user_id}",
    "security.model_encrypted": "Model encrypted successfully",
    "security.integrity_check_passed": "Model integrity check passed",
    "security.integrity_check_failed": "Model integrity check failed",
    
    # Monitoring
    "monitoring.alert": "Alert: {metric} = {value} exceeds threshold {threshold}",
    "monitoring.health_check": "System health: {status}",
    "monitoring.metrics_collected": "Collected {count} metrics",
    
    # Errors
    "error.invalid_input": "Invalid input data provided",
    "error.model_not_found": "Model file not found: {path}",
    "error.insufficient_memory": "Insufficient memory for operation",
    "error.gpu_not_available": "GPU not available for computation",
    "error.connection_failed": "Connection failed to {service}",
    
    # Research features
    "research.causal_discovery": "Causal discovery completed - {edges} causal edges found",
    "research.quantum_processing": "Quantum processing completed - coherence: {coherence:.3f}",
    "research.federated_round": "Federated round {round} - {clients} clients participated",
    
    # Deployment
    "deploy.edge_optimized": "Model optimized for edge deployment - size: {size:.1f} MB",
    "deploy.cloud_ready": "Cloud deployment ready - {regions} regions configured",
    "deploy.scaling": "Auto-scaling triggered - new capacity: {instances} instances"
}

# Spanish
MESSAGES_ES = {
    # Training messages
    "training.started": "Entrenamiento iniciado con {epochs} épocas",
    "training.epoch_complete": "Época {epoch}/{total_epochs} completada - Pérdida: {loss:.4f}",
    "training.early_stopping": "Parada temprana activada en época {epoch}",
    "training.completed": "Entrenamiento completado exitosamente",
    "training.failed": "Entrenamiento falló: {error}",
    
    # Model messages
    "model.created": "Modelo DGDN creado con {parameters:,} parámetros",
    "model.loaded": "Modelo cargado desde {path}",
    "model.saved": "Modelo guardado en {path}",
    "model.validation_failed": "Validación del modelo falló: {error}",
    
    # Data processing
    "data.loaded": "Conjunto de datos cargado: {nodes} nodos, {edges} aristas",
    "data.preprocessing": "Preprocesando datos de grafo temporal...",
    "data.validation_error": "Error de validación de datos: {error}",
    
    # Performance
    "perf.inference_time": "Inferencia completada en {time:.3f}s",
    "perf.memory_usage": "Uso de memoria: {memory:.1f} MB",
    "perf.throughput": "Rendimiento: {rps:.1f} solicitudes/segundo",
    
    # Security
    "security.access_granted": "Acceso concedido al usuario {user_id}",
    "security.access_denied": "Acceso denegado para usuario {user_id}",
    "security.model_encrypted": "Modelo encriptado exitosamente",
    "security.integrity_check_passed": "Verificación de integridad del modelo exitosa",
    "security.integrity_check_failed": "Verificación de integridad del modelo falló",
    
    # Monitoring
    "monitoring.alert": "Alerta: {metric} = {value} excede el umbral {threshold}",
    "monitoring.health_check": "Salud del sistema: {status}",
    "monitoring.metrics_collected": "Recolectadas {count} métricas",
    
    # Errors
    "error.invalid_input": "Datos de entrada inválidos proporcionados",
    "error.model_not_found": "Archivo de modelo no encontrado: {path}",
    "error.insufficient_memory": "Memoria insuficiente para la operación",
    "error.gpu_not_available": "GPU no disponible para computación",
    "error.connection_failed": "Conexión falló a {service}",
    
    # Research features
    "research.causal_discovery": "Descubrimiento causal completado - {edges} aristas causales encontradas",
    "research.quantum_processing": "Procesamiento cuántico completado - coherencia: {coherence:.3f}",
    "research.federated_round": "Ronda federada {round} - {clients} clientes participaron",
    
    # Deployment
    "deploy.edge_optimized": "Modelo optimizado para despliegue edge - tamaño: {size:.1f} MB",
    "deploy.cloud_ready": "Despliegue en nube listo - {regions} regiones configuradas",
    "deploy.scaling": "Auto-escalado activado - nueva capacidad: {instances} instancias"
}

# French
MESSAGES_FR = {
    # Training messages
    "training.started": "Entraînement commencé avec {epochs} époques",
    "training.epoch_complete": "Époque {epoch}/{total_epochs} terminée - Perte: {loss:.4f}",
    "training.early_stopping": "Arrêt précoce déclenché à l'époque {epoch}",
    "training.completed": "Entraînement terminé avec succès",
    "training.failed": "Entraînement échoué: {error}",
    
    # Model messages
    "model.created": "Modèle DGDN créé avec {parameters:,} paramètres",
    "model.loaded": "Modèle chargé depuis {path}",
    "model.saved": "Modèle sauvegardé vers {path}",
    "model.validation_failed": "Validation du modèle échouée: {error}",
    
    # Data processing
    "data.loaded": "Jeu de données chargé: {nodes} nœuds, {edges} arêtes",
    "data.preprocessing": "Prétraitement des données de graphe temporel...",
    "data.validation_error": "Erreur de validation des données: {error}",
    
    # Performance
    "perf.inference_time": "Inférence terminée en {time:.3f}s",
    "perf.memory_usage": "Usage mémoire: {memory:.1f} MB",
    "perf.throughput": "Débit: {rps:.1f} requêtes/seconde",
    
    # Security
    "security.access_granted": "Accès accordé à l'utilisateur {user_id}",
    "security.access_denied": "Accès refusé pour l'utilisateur {user_id}",
    "security.model_encrypted": "Modèle chiffré avec succès",
    "security.integrity_check_passed": "Vérification d'intégrité du modèle réussie",
    "security.integrity_check_failed": "Vérification d'intégrité du modèle échouée",
    
    # Monitoring
    "monitoring.alert": "Alerte: {metric} = {value} dépasse le seuil {threshold}",
    "monitoring.health_check": "Santé du système: {status}",
    "monitoring.metrics_collected": "Collecté {count} métriques",
    
    # Errors
    "error.invalid_input": "Données d'entrée invalides fournies",
    "error.model_not_found": "Fichier de modèle non trouvé: {path}",
    "error.insufficient_memory": "Mémoire insuffisante pour l'opération",
    "error.gpu_not_available": "GPU non disponible pour le calcul",
    "error.connection_failed": "Connexion échouée vers {service}",
    
    # Research features
    "research.causal_discovery": "Découverte causale terminée - {edges} arêtes causales trouvées",
    "research.quantum_processing": "Traitement quantique terminé - cohérence: {coherence:.3f}",
    "research.federated_round": "Ronde fédérée {round} - {clients} clients ont participé",
    
    # Deployment
    "deploy.edge_optimized": "Modèle optimisé pour déploiement edge - taille: {size:.1f} MB",
    "deploy.cloud_ready": "Déploiement cloud prêt - {regions} régions configurées",
    "deploy.scaling": "Auto-dimensionnement déclenché - nouvelle capacité: {instances} instances"
}

# German
MESSAGES_DE = {
    # Training messages
    "training.started": "Training begonnen mit {epochs} Epochen",
    "training.epoch_complete": "Epoche {epoch}/{total_epochs} abgeschlossen - Verlust: {loss:.4f}",
    "training.early_stopping": "Frühes Stoppen bei Epoche {epoch} ausgelöst",
    "training.completed": "Training erfolgreich abgeschlossen",
    "training.failed": "Training fehlgeschlagen: {error}",
    
    # Model messages
    "model.created": "DGDN-Modell mit {parameters:,} Parametern erstellt",
    "model.loaded": "Modell geladen von {path}",
    "model.saved": "Modell gespeichert nach {path}",
    "model.validation_failed": "Modellvalidierung fehlgeschlagen: {error}",
    
    # Data processing
    "data.loaded": "Datensatz geladen: {nodes} Knoten, {edges} Kanten",
    "data.preprocessing": "Vorverarbeitung der temporalen Graphdaten...",
    "data.validation_error": "Datenvalidierungsfehler: {error}",
    
    # Performance
    "perf.inference_time": "Inferenz abgeschlossen in {time:.3f}s",
    "perf.memory_usage": "Speicherverbrauch: {memory:.1f} MB",
    "perf.throughput": "Durchsatz: {rps:.1f} Anfragen/Sekunde",
    
    # Security
    "security.access_granted": "Zugriff gewährt für Benutzer {user_id}",
    "security.access_denied": "Zugriff verweigert für Benutzer {user_id}",
    "security.model_encrypted": "Modell erfolgreich verschlüsselt",
    "security.integrity_check_passed": "Modell-Integritätsprüfung bestanden",
    "security.integrity_check_failed": "Modell-Integritätsprüfung fehlgeschlagen",
    
    # Monitoring
    "monitoring.alert": "Warnung: {metric} = {value} überschreitet Schwellenwert {threshold}",
    "monitoring.health_check": "Systemzustand: {status}",
    "monitoring.metrics_collected": "{count} Metriken gesammelt",
    
    # Errors
    "error.invalid_input": "Ungültige Eingabedaten bereitgestellt",
    "error.model_not_found": "Modelldatei nicht gefunden: {path}",
    "error.insufficient_memory": "Unzureichender Speicher für Operation",
    "error.gpu_not_available": "GPU nicht verfügbar für Berechnung",
    "error.connection_failed": "Verbindung zu {service} fehlgeschlagen",
    
    # Research features
    "research.causal_discovery": "Kausale Entdeckung abgeschlossen - {edges} kausale Kanten gefunden",
    "research.quantum_processing": "Quantenverarbeitung abgeschlossen - Kohärenz: {coherence:.3f}",
    "research.federated_round": "Föderierte Runde {round} - {clients} Clients teilgenommen",
    
    # Deployment
    "deploy.edge_optimized": "Modell für Edge-Deployment optimiert - Größe: {size:.1f} MB",
    "deploy.cloud_ready": "Cloud-Deployment bereit - {regions} Regionen konfiguriert",
    "deploy.scaling": "Auto-Skalierung ausgelöst - neue Kapazität: {instances} Instanzen"
}

# Japanese
MESSAGES_JA = {
    # Training messages
    "training.started": "{epochs}エポックでトレーニングを開始",
    "training.epoch_complete": "エポック {epoch}/{total_epochs} 完了 - 損失: {loss:.4f}",
    "training.early_stopping": "エポック{epoch}で早期停止が発動",
    "training.completed": "トレーニングが正常に完了",
    "training.failed": "トレーニングが失敗: {error}",
    
    # Model messages
    "model.created": "{parameters:,}パラメータでDGDNモデルを作成",
    "model.loaded": "{path}からモデルを読み込み",
    "model.saved": "{path}にモデルを保存",
    "model.validation_failed": "モデル検証が失敗: {error}",
    
    # Data processing
    "data.loaded": "データセットを読み込み: {nodes}ノード, {edges}エッジ",
    "data.preprocessing": "時間的グラフデータを前処理中...",
    "data.validation_error": "データ検証エラー: {error}",
    
    # Performance
    "perf.inference_time": "{time:.3f}秒で推論完了",
    "perf.memory_usage": "メモリ使用量: {memory:.1f} MB",
    "perf.throughput": "スループット: {rps:.1f} リクエスト/秒",
    
    # Security
    "security.access_granted": "ユーザー{user_id}にアクセス許可",
    "security.access_denied": "ユーザー{user_id}のアクセス拒否",
    "security.model_encrypted": "モデルの暗号化に成功",
    "security.integrity_check_passed": "モデル整合性チェック合格",
    "security.integrity_check_failed": "モデル整合性チェック失敗",
    
    # Monitoring
    "monitoring.alert": "アラート: {metric} = {value} が閾値 {threshold} を超過",
    "monitoring.health_check": "システムヘルス: {status}",
    "monitoring.metrics_collected": "{count}個のメトリクスを収集",
    
    # Errors
    "error.invalid_input": "無効な入力データが提供されました",
    "error.model_not_found": "モデルファイルが見つかりません: {path}",
    "error.insufficient_memory": "操作に必要なメモリが不足",
    "error.gpu_not_available": "計算用GPUが利用不可",
    "error.connection_failed": "{service}への接続が失敗",
    
    # Research features
    "research.causal_discovery": "因果発見完了 - {edges}個の因果エッジを発見",
    "research.quantum_processing": "量子処理完了 - コヒーレンス: {coherence:.3f}",
    "research.federated_round": "連合ラウンド{round} - {clients}クライアント参加",
    
    # Deployment
    "deploy.edge_optimized": "エッジデプロイ用に最適化 - サイズ: {size:.1f} MB",
    "deploy.cloud_ready": "クラウドデプロイ準備完了 - {regions}リージョン設定済み",
    "deploy.scaling": "オートスケーリング発動 - 新容量: {instances}インスタンス"
}

# Chinese (Simplified)
MESSAGES_ZH = {
    # Training messages
    "training.started": "开始训练，共{epochs}个轮次",
    "training.epoch_complete": "轮次 {epoch}/{total_epochs} 完成 - 损失: {loss:.4f}",
    "training.early_stopping": "在第{epoch}轮次触发早停",
    "training.completed": "训练成功完成",
    "training.failed": "训练失败: {error}",
    
    # Model messages
    "model.created": "创建DGDN模型，共{parameters:,}个参数",
    "model.loaded": "从{path}加载模型",
    "model.saved": "模型保存到{path}",
    "model.validation_failed": "模型验证失败: {error}",
    
    # Data processing
    "data.loaded": "数据集已加载: {nodes}个节点, {edges}条边",
    "data.preprocessing": "正在预处理时序图数据...",
    "data.validation_error": "数据验证错误: {error}",
    
    # Performance
    "perf.inference_time": "推理在{time:.3f}秒内完成",
    "perf.memory_usage": "内存使用量: {memory:.1f} MB",
    "perf.throughput": "吞吐量: {rps:.1f} 请求/秒",
    
    # Security
    "security.access_granted": "用户{user_id}访问授权",
    "security.access_denied": "用户{user_id}访问拒绝",
    "security.model_encrypted": "模型加密成功",
    "security.integrity_check_passed": "模型完整性检查通过",
    "security.integrity_check_failed": "模型完整性检查失败",
    
    # Monitoring
    "monitoring.alert": "警报: {metric} = {value} 超过阈值 {threshold}",
    "monitoring.health_check": "系统健康状态: {status}",
    "monitoring.metrics_collected": "收集了{count}项指标",
    
    # Errors
    "error.invalid_input": "提供了无效的输入数据",
    "error.model_not_found": "未找到模型文件: {path}",
    "error.insufficient_memory": "操作内存不足",
    "error.gpu_not_available": "GPU不可用于计算",
    "error.connection_failed": "连接{service}失败",
    
    # Research features
    "research.causal_discovery": "因果发现完成 - 发现{edges}条因果边",
    "research.quantum_processing": "量子处理完成 - 相干性: {coherence:.3f}",
    "research.federated_round": "联邦轮次{round} - {clients}个客户端参与",
    
    # Deployment
    "deploy.edge_optimized": "边缘部署优化完成 - 大小: {size:.1f} MB",
    "deploy.cloud_ready": "云部署就绪 - 配置了{regions}个区域",
    "deploy.scaling": "自动扩展触发 - 新容量: {instances}个实例"
}

# Message registry
ALL_MESSAGES = {
    'en': MESSAGES_EN,
    'es': MESSAGES_ES,
    'fr': MESSAGES_FR,
    'de': MESSAGES_DE,
    'ja': MESSAGES_JA,
    'zh': MESSAGES_ZH
}

# Supported locales
SUPPORTED_LOCALES = list(ALL_MESSAGES.keys())

# Default locale
DEFAULT_LOCALE = 'en'