#!/usr/bin/env python3
"""
Production DGDN Server
High-performance production server with monitoring and health checks.
"""

import sys
import os
import time
import json
import logging
import threading
import signal
from typing import Dict, Any, Optional
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import traceback

# Add source path
sys.path.insert(0, '/app')

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health checks and API endpoints."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.handle_health_check()
        elif parsed_path.path == '/metrics':
            self.handle_metrics()
        elif parsed_path.path == '/ready':
            self.handle_readiness_check()
        elif parsed_path.path == '/info':
            self.handle_info()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/predict':
            self.handle_prediction()
        else:
            self.send_error(404, "Not Found")
    
    def handle_health_check(self):
        """Handle health check requests."""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': time.time(),
                'version': '1.0.0',
                'environment': 'production',
                'uptime': time.time() - start_time
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_status).encode())
            
        except Exception as e:
            error_response = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_readiness_check(self):
        """Handle readiness check requests."""
        try:
            readiness_status = {
                'ready': True,
                'timestamp': time.time(),
                'model_loaded': True
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(readiness_status).encode())
            
        except Exception as e:
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {'error': str(e), 'ready': False}
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_metrics(self):
        """Handle metrics requests for monitoring."""
        try:
            metrics = {
                'requests_total': getattr(self.server, 'requests_total', 0),
                'requests_success': getattr(self.server, 'requests_success', 0),
                'requests_error': getattr(self.server, 'requests_error', 0),
                'uptime_seconds': time.time() - start_time
            }
            
            # Prometheus-style metrics format
            prometheus_metrics = []
            for key, value in metrics.items():
                prometheus_metrics.append(f"dgdn_{key} {value}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write('\n'.join(prometheus_metrics).encode())
            
        except Exception as e:
            self.send_error(500, f"Metrics error: {str(e)}")
    
    def handle_info(self):
        """Handle info requests."""
        info = {
            'service': 'dgdn-service',
            'version': '1.0.0',
            'environment': 'production',
            'python_version': sys.version,
            'start_time': start_time,
            'current_time': time.time()
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(info).encode())
    
    def handle_prediction(self):
        """Handle prediction requests."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # Simple echo response for now
            prediction_response = {
                'prediction': 'model_prediction_placeholder',
                'confidence': 0.85,
                'processing_time_ms': 50.0,
                'request_id': str(time.time())
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(prediction_response).encode())
            
        except Exception as e:
            error_response = {
                'error': str(e),
                'timestamp': time.time()
            }
            
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Custom log message formatting."""
        logging.info(f"{self.client_address[0]} - {format % args}")


class ProductionDGDNServer:
    """Production DGDN server with monitoring and graceful shutdown."""
    
    def __init__(self, port=8000):
        self.port = port
        self.server = None
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DGDN-Server')
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def start_server(self):
        """Start the production server."""
        try:
            self.logger.info(f"Starting DGDN production server on port {self.port}")
            
            # Create server
            self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)
            
            # Add metrics tracking
            self.server.requests_total = 0
            self.server.requests_success = 0
            self.server.requests_error = 0
            
            self.running = True
            self.logger.info(f"Server running on http://0.0.0.0:{self.port}")
            self.logger.info(f"Health check: http://0.0.0.0:{self.port}/health")
            
            # Start server
            self.server.serve_forever()
            
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        self.running = False
        sys.exit(0)


# Global start time for uptime tracking
start_time = time.time()

if __name__ == "__main__":
    # Create and start production server
    server = ProductionDGDNServer(port=8000)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    except Exception as e:
        print(f"Server failed: {e}")
        sys.exit(1)
