"""
Load balancing and request routing for DGDN deployments.
"""

import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    RESPONSE_TIME = "response_time"

@dataclass
class ServerInstance:
    """Server instance information."""
    id: str
    endpoint: str
    weight: float = 1.0
    active_connections: int = 0
    avg_response_time: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0

class LoadBalancer:
    """Intelligent load balancer for DGDN model servers."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.servers = []
        self.current_index = 0
        self.lock = threading.Lock()
    
    def add_server(self, server: ServerInstance):
        """Add server to the pool."""
        with self.lock:
            self.servers.append(server)
    
    def remove_server(self, server_id: str):
        """Remove server from the pool."""
        with self.lock:
            self.servers = [s for s in self.servers if s.id != server_id]
    
    def get_server(self) -> Optional[ServerInstance]:
        """Get next server based on strategy."""
        with self.lock:
            healthy_servers = [s for s in self.servers if s.is_healthy]
            
            if not healthy_servers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin(healthy_servers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections(healthy_servers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
                return self._weighted_random(healthy_servers)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._response_time_based(healthy_servers)
            
            return healthy_servers[0]  # Fallback
    
    def _round_robin(self, servers: List[ServerInstance]) -> ServerInstance:
        """Round-robin server selection."""
        server = servers[self.current_index % len(servers)]
        self.current_index = (self.current_index + 1) % len(servers)
        return server
    
    def _least_connections(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server with least active connections."""
        return min(servers, key=lambda s: s.active_connections)
    
    def _weighted_random(self, servers: List[ServerInstance]) -> ServerInstance:
        """Weighted random server selection."""
        weights = [s.weight for s in servers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(servers)
        
        r = random.uniform(0, total_weight)
        current_weight = 0
        
        for server, weight in zip(servers, weights):
            current_weight += weight
            if r <= current_weight:
                return server
        
        return servers[-1]  # Fallback
    
    def _response_time_based(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server with best response time."""
        return min(servers, key=lambda s: s.avg_response_time or float('inf'))

class RequestRouter:
    """Intelligent request router for different model types."""
    
    def __init__(self):
        self.model_pools = {}
        self.routing_rules = []
    
    def add_model_pool(self, model_type: str, load_balancer: LoadBalancer):
        """Add model pool for specific type."""
        self.model_pools[model_type] = load_balancer
    
    def route_request(self, request: Dict[str, Any]) -> Optional[ServerInstance]:
        """Route request to appropriate model pool."""
        model_type = self._determine_model_type(request)
        
        if model_type not in self.model_pools:
            return None
        
        return self.model_pools[model_type].get_server()
    
    def _determine_model_type(self, request: Dict[str, Any]) -> str:
        """Determine which model type to use for request."""
        # Simple routing logic (can be enhanced)
        if request.get('num_nodes', 0) > 1000:
            return 'large_model'
        elif request.get('real_time', False):
            return 'fast_model'
        else:
            return 'standard_model'