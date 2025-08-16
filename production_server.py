#!/usr/bin/env python3
"""
Production DGDN Server

High-performance production server with monitoring, health checks,
and enterprise-grade features.
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Add src to path
sys.path.insert(0, 'src')

import dgdn
from dgdn import DynamicGraphDiffusionNet, TemporalData


# Metrics
INFERENCE_COUNTER = Counter('dgdn_inferences_total', 'Total number of inferences')
INFERENCE_TIME = Histogram('dgdn_inference_time_seconds', 'Inference time in seconds')
CACHE_HIT_RATE = Gauge('dgdn_cache_hit_rate', 'Cache hit rate')
MEMORY_USAGE = Gauge('dgdn_memory_usage_bytes', 'Memory usage in bytes')
ERROR_COUNTER = Counter('dgdn_errors_total', 'Total number of errors', ['error_type'])


class GraphData(BaseModel):
    """Input graph data schema."""
    edge_index: list = Field(..., description="Edge connectivity as list of [source, target] pairs")
    timestamps: list = Field(..., description="Edge timestamps")
    node_features: Optional[list] = Field(None, description="Node features matrix")
    edge_attr: Optional[list] = Field(None, description="Edge attributes matrix")
    num_nodes: int = Field(..., description="Number of nodes in graph")


class InferenceRequest(BaseModel):
    """Inference request schema."""
    graph_data: GraphData
    return_attention: bool = Field(False, description="Return attention weights")
    return_uncertainty: bool = Field(False, description="Return uncertainty estimates")
    use_cache: bool = Field(True, description="Use caching if available")


class InferenceResponse(BaseModel):
    """Inference response schema."""
    node_embeddings: list
    inference_time: float
    cache_hit: bool
    model_version: str
    timestamp: str


class ProductionDGDNServer:
    """Production DGDN server implementation."""
    
    def __init__(self):
        self.model = None
        self.cache = {}
        self.startup_time = datetime.now()
        self.request_count = 0
        
        # Load configuration
        self.config = {
            "model_path": os.getenv("DGDN_MODEL_PATH", "models/dgdn_model.pth"),
            "cache_size": int(os.getenv("DGDN_CACHE_SIZE", "1024")),
            "max_workers": int(os.getenv("DGDN_MAX_WORKERS", "4")),
            "log_level": os.getenv("DGDN_LOG_LEVEL", "INFO")
        }
        
        self._setup_logging()
        self._initialize_model()
    
    def _setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('/app/logs/dgdn_server.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 DGDN Production Server starting...")
    
    def _initialize_model(self):
        """Initialize the DGDN model."""
        try:
            self.model = DynamicGraphDiffusionNet(
                node_dim=128,
                edge_dim=64, 
                hidden_dim=256,
                num_layers=3,
                num_heads=8,
                diffusion_steps=5,
                dropout=0.0
            )
            self.model.eval()
            
            # Load pre-trained weights if available
            model_path = Path(self.config["model_path"])
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.logger.info(f"📦 Model loaded from {model_path}")
            else:
                self.logger.warning(f"⚠️ Model file not found at {model_path}, using random weights")
            
            self.logger.info("✅ DGDN model initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def convert_to_temporal_data(self, graph_data: GraphData) -> TemporalData:
        """Convert input data to TemporalData format."""
        edge_index = torch.tensor(graph_data.edge_index).t().contiguous()
        timestamps = torch.tensor(graph_data.timestamps, dtype=torch.float32)
        
        node_features = None
        if graph_data.node_features:
            node_features = torch.tensor(graph_data.node_features, dtype=torch.float32)
        
        edge_attr = None
        if graph_data.edge_attr:
            edge_attr = torch.tensor(graph_data.edge_attr, dtype=torch.float32)
        
        return TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=node_features,
            edge_attr=edge_attr,
            num_nodes=graph_data.num_nodes
        )
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform model inference."""
        start_time = time.time()
        cache_hit = False
        
        try:
            # Convert input data
            data = self.convert_to_temporal_data(request.graph_data)
            
            # Check cache
            cache_key = None
            if request.use_cache:
                # Simple cache key based on data hash
                cache_key = hash(str(request.graph_data.dict()))
                if cache_key in self.cache:
                    result = self.cache[cache_key]
                    cache_hit = True
                    CACHE_HIT_RATE.set(len([v for v in self.cache.values() if v]) / max(len(self.cache), 1))
                    self.logger.info(f"💾 Cache hit for request")
                else:
                    # Perform inference
                    with torch.no_grad():
                        output = self.model(
                            data,
                            return_attention=request.return_attention,
                            return_uncertainty=request.return_uncertainty
                        )
                        result = output["node_embeddings"].tolist()
                    
                    # Cache result
                    if len(self.cache) < self.config["cache_size"]:
                        self.cache[cache_key] = result
            else:
                # Perform inference without caching
                with torch.no_grad():
                    output = self.model(
                        data,
                        return_attention=request.return_attention,
                        return_uncertainty=request.return_uncertainty
                    )
                    result = output["node_embeddings"].tolist()
            
            inference_time = time.time() - start_time
            
            # Update metrics
            INFERENCE_COUNTER.inc()
            INFERENCE_TIME.observe(inference_time)
            MEMORY_USAGE.set(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
            
            self.request_count += 1
            
            response = InferenceResponse(
                node_embeddings=result,
                inference_time=inference_time,
                cache_hit=cache_hit,
                model_version=dgdn.__version__,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"✅ Inference completed in {inference_time:.3f}s (cache_hit={cache_hit})")
            return response
            
        except Exception as e:
            ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
            self.logger.error(f"❌ Inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Global server instance
server = ProductionDGDNServer()

# FastAPI app
app = FastAPI(
    title="DGDN Production API",
    description="Dynamic Graph Diffusion Network - Production API",
    version=dgdn.__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - server.startup_time).total_seconds()
    
    return {
        "status": "healthy",
        "version": dgdn.__version__,
        "uptime_seconds": uptime,
        "requests_served": server.request_count,
        "model_loaded": server.model is not None,
        "cache_size": len(server.cache),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(prometheus_client.REGISTRY)


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """Main inference endpoint."""
    return await server.inference(request)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "DGDN Production API",
        "version": dgdn.__version__,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "production_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for simplicity, use gunicorn for multiple workers
        log_level="info",
        access_log=True
    )
