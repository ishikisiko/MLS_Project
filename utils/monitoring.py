"""
Prometheus Monitoring Module for Federated Learning System.

This module provides:
1. Server Metrics: Training rounds, connected clients, aggregation time, loss
2. Client Metrics: Training duration, memory/CPU usage
3. HTTP metrics endpoint for Prometheus scraping
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, REGISTRY, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Monitoring disabled.")
    print("Install with: pip install prometheus_client")


@dataclass
class ClientMetrics:
    """Stores metrics for a single client."""
    client_id: str
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    training_duration_seconds: float = 0.0
    current_round: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


class FederatedLearningMetrics:
    """
    Prometheus metrics collector for Federated Learning system.
    
    Usage on Server:
        metrics = FederatedLearningMetrics()
        metrics.start_http_server(port=8000)
        
        # During training
        metrics.set_round(round_number)
        metrics.record_aggregation_time(duration)
        metrics.set_loss(loss_value)
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            return
            
        self._enabled = True
        self._registry = registry or REGISTRY
        self._client_metrics: Dict[str, ClientMetrics] = {}
        self._lock = threading.Lock()
        
        # ============ Server Metrics ============
        
        # Training round counter
        self.round_gauge = Gauge(
            'fl_round_current',
            'Current federated learning round number',
            registry=self._registry
        )
        
        # Total rounds completed
        self.rounds_total = Counter(
            'fl_rounds_total',
            'Total number of federated learning rounds completed',
            registry=self._registry
        )
        
        # Connected clients
        self.connected_clients = Gauge(
            'fl_connected_clients',
            'Number of currently connected clients',
            registry=self._registry
        )
        
        # Waiting updates
        self.waiting_updates = Gauge(
            'fl_waiting_updates',
            'Number of client updates waiting for aggregation',
            registry=self._registry
        )
        
        # Total training examples
        self.total_examples = Counter(
            'fl_training_examples_total',
            'Total number of training examples processed',
            registry=self._registry
        )
        
        # Aggregation time histogram
        self.aggregation_duration = Histogram(
            'fl_aggregation_duration_seconds',
            'Time spent aggregating client updates',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self._registry
        )
        
        # Average loss
        self.avg_loss = Gauge(
            'fl_avg_loss',
            'Average loss from last aggregation',
            registry=self._registry
        )
        
        # Model size
        self.model_size_mb = Gauge(
            'fl_model_size_mb',
            'Size of the global model in MB',
            registry=self._registry
        )
        
        # ============ Client Status Metrics ============
        
        # Per-client memory usage (labeled by client_id)
        self.client_memory = Gauge(
            'fl_client_memory_usage_mb',
            'Client memory usage in MB',
            ['client_id'],
            registry=self._registry
        )
        
        # Per-client CPU usage
        self.client_cpu = Gauge(
            'fl_client_cpu_percent',
            'Client CPU usage percentage',
            ['client_id'],
            registry=self._registry
        )
        
        # Per-client training duration
        self.client_training_time = Gauge(
            'fl_client_training_duration_seconds',
            'Client training duration for last round',
            ['client_id'],
            registry=self._registry
        )
        
        # Server start time
        self.server_start_time = Gauge(
            'fl_server_start_timestamp',
            'Server start time as Unix timestamp',
            registry=self._registry
        )
        self.server_start_time.set_to_current_time()
        
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def start_http_server(self, port: int = 8000):
        """Start the Prometheus metrics HTTP endpoint."""
        if not self._enabled:
            print("Monitoring disabled, skipping HTTP server start")
            return
            
        try:
            start_http_server(port, registry=self._registry)
            print(f"✓ Prometheus metrics available at http://localhost:{port}/metrics")
        except Exception as e:
            print(f"Failed to start metrics server: {e}")
    
    # ============ Server Metric Methods ============
    
    def set_round(self, round_number: int):
        """Update the current round number."""
        if self._enabled:
            self.round_gauge.set(round_number)
    
    def increment_round(self):
        """Increment round counter when a round completes."""
        if self._enabled:
            self.rounds_total.inc()
    
    def set_connected_clients(self, count: int):
        """Set the number of connected clients."""
        if self._enabled:
            self.connected_clients.set(count)
    
    def set_waiting_updates(self, count: int):
        """Set the number of waiting updates."""
        if self._enabled:
            self.waiting_updates.set(count)
    
    def add_examples(self, count: int):
        """Add to the total examples counter."""
        if self._enabled:
            self.total_examples.inc(count)
    
    def record_aggregation_time(self, duration_seconds: float):
        """Record aggregation duration."""
        if self._enabled:
            self.aggregation_duration.observe(duration_seconds)
    
    def set_loss(self, loss: float):
        """Set the average loss value."""
        if self._enabled:
            self.avg_loss.set(loss)
    
    def set_model_size(self, size_mb: float):
        """Set the model size in MB."""
        if self._enabled:
            self.model_size_mb.set(size_mb)
    
    # ============ Client Status Methods ============
    
    def update_client_status(self, client_id: str, memory_mb: float = 0.0,
                            cpu_percent: float = 0.0, 
                            training_duration: float = 0.0,
                            current_round: int = 0):
        """Update metrics for a specific client."""
        if not self._enabled:
            return
            
        with self._lock:
            self._client_metrics[client_id] = ClientMetrics(
                client_id=client_id,
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent,
                training_duration_seconds=training_duration,
                current_round=current_round,
                last_seen=datetime.now()
            )
            
        # Update Prometheus gauges
        safe_id = client_id.replace(':', '_').replace('[', '').replace(']', '')
        self.client_memory.labels(client_id=safe_id).set(memory_mb)
        self.client_cpu.labels(client_id=safe_id).set(cpu_percent)
        self.client_training_time.labels(client_id=safe_id).set(training_duration)
    
    def get_client_status(self, client_id: str) -> Optional[ClientMetrics]:
        """Get the current status of a client."""
        with self._lock:
            return self._client_metrics.get(client_id)
    
    def get_all_clients(self) -> Dict[str, ClientMetrics]:
        """Get all client metrics."""
        with self._lock:
            return dict(self._client_metrics)
    
    def remove_client(self, client_id: str):
        """Remove a client from tracking."""
        with self._lock:
            if client_id in self._client_metrics:
                del self._client_metrics[client_id]


class AggregationTimer:
    """Context manager for timing aggregation operations."""
    
    def __init__(self, metrics: FederatedLearningMetrics):
        self.metrics = metrics
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None and self.metrics.enabled:
            duration = time.time() - self.start_time
            self.metrics.record_aggregation_time(duration)
        return False


# Global metrics instance (singleton pattern for easy access)
_global_metrics: Optional[FederatedLearningMetrics] = None


def get_metrics() -> FederatedLearningMetrics:
    """Get or create the global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = FederatedLearningMetrics()
    return _global_metrics


def init_metrics(port: int = 8000) -> FederatedLearningMetrics:
    """Initialize and start the metrics server."""
    metrics = get_metrics()
    metrics.start_http_server(port)
    return metrics


# ============ Utility Functions ============

def format_metrics_summary(metrics: FederatedLearningMetrics) -> str:
    """Generate a text summary of current metrics."""
    if not metrics.enabled:
        return "Monitoring disabled"
        
    clients = metrics.get_all_clients()
    lines = [
        "=== Federated Learning Metrics ===",
        f"Active Clients: {len(clients)}",
        ""
    ]
    
    if clients:
        lines.append("Client Status:")
        for cid, cm in clients.items():
            lines.append(
                f"  {cid[:20]}... | Mem: {cm.memory_usage_mb:.1f}MB | "
                f"CPU: {cm.cpu_percent:.1f}% | Train: {cm.training_duration_seconds:.2f}s"
            )
    
    return "\n".join(lines)
