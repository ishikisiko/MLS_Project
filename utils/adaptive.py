"""
Adaptive System Behavior Module.

This module provides functionalities for:
1. Network Monitoring: Estimating bandwidth, latency, and network stability.
2. Device Availability: Checking battery, memory pressure, and CPU load.
3. Adaptive Strategy: Making decisions on participation and compression based on context.
"""

import time
import os
import random
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from utils import config

@dataclass
class AdaptiveParams:
    """Parameters derived from adaptive strategy."""
    should_participate: bool
    compression_level: str # 'none', 'light', 'heavy'
    local_epochs: int
    batch_size: int
    upload_strategy: Dict[str, any]

class NetworkMonitor:
    """
    Monitors network conditions (latency, bandwidth, stability).
    In a real system, this would perform active probing or passive observation.
    Here we simulate some aspects and use simple measurements.
    """
    def __init__(self, server_host: str = "localhost"):
        self.server_host = server_host
        self.latency_history = []
        self.last_check = 0
        self.cached_score = 100.0
        
    def measure_latency(self) -> float:
        """
        Measure ping latency to server.
        Returns latency in milliseconds.
        """
        # In a real deployment, we might use 'ping' command or send a small packet.
        # For this simulation/prototype, we'll simulate it or use a simple heuristic.
        
        # Simulation: Base latency + Jitter
        base_latency = 20.0 # ms
        jitter = random.uniform(0, 10.0) 
        
        # Inject occasional spikes to simulate network instability
        if random.random() < 0.1:
            jitter += 100.0
            
        latency = base_latency + jitter
        
        self.latency_history.append(latency)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
            
        return latency

    def estimate_bandwidth(self) -> float:
        """
        Estimate available upload bandwidth in Mbps.
        """
        # Placeholder for real bandwidth estimation (e.g., packet train).
        # We simulate a value between 1 and 20 Mbps.
        return random.uniform(1.0, 20.0)

    def is_network_stable(self) -> bool:
        """
        Check if network is stable based on latency variance.
        """
        if len(self.latency_history) < 3:
            return True
            
        # Calculate variance
        avg = sum(self.latency_history) / len(self.latency_history)
        variance = sum((x - avg) ** 2 for x in self.latency_history) / len(self.latency_history)
        std_dev = math.sqrt(variance)
        
        # If jitter is high (> 30ms), consider unstable
        return std_dev < 30.0

    def get_network_quality_score(self) -> float:
        """
        Calculate a 0-100 score representing overall network quality.
        Higher is better.
        """
        current_time = time.time()
        if current_time - self.last_check < 2.0: # Cache for 2 seconds
            return self.cached_score
            
        latency = self.measure_latency()
        bandwidth = self.estimate_bandwidth()
        stable = self.is_network_stable()
        
        # Scoring Logic:
        # Latency: < 50ms -> 100, > 500ms -> 0
        latency_score = max(0, 100 - (latency - 20) / 4.8) # Approx map
        
        # Bandwidth: > 10Mbps -> 100, < 1Mbps -> 0
        bw_score = min(100, max(0, bandwidth * 10))
        
        # Stability penalty
        stability_factor = 1.0 if stable else 0.5
        
        final_score = (0.6 * latency_score + 0.4 * bw_score) * stability_factor
        
        self.cached_score = final_score
        self.last_check = current_time
        return final_score


class DeviceAvailability:
    """
    Checks device availability (Memory, CPU, Battery).
    """
    def __init__(self):
        self.simulated_battery = 100.0 # Start full
        self.last_check = time.time()
        
    def _update_simulated_battery(self):
        """Simulate battery drain."""
        now = time.time()
        elapsed = now - self.last_check
        # Drain 1% every 60 seconds of active work
        drain = (elapsed / 60.0) * 1.0
        self.simulated_battery = max(0.0, self.simulated_battery - drain)
        self.last_check = now

    def check_memory_pressure(self) -> str:
        """Returns 'low', 'medium', 'high'."""
        if not PSUTIL_AVAILABLE:
            return 'low'
            
        vm = psutil.virtual_memory()
        percent = vm.percent
        
        if percent > 85:
            return 'high'
        elif percent > 60:
            return 'medium'
        else:
            return 'low'

    def get_cpu_load(self) -> float:
        """Returns CPU usage percentage."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        return psutil.cpu_percent(interval=None)

    def estimate_battery_level(self) -> float:
        """
        Get battery level (0-100).
        If psutil sensors not available (common on desktops/server docker), simulate it.
        """
        self._update_simulated_battery()
        
        if PSUTIL_AVAILABLE and hasattr(psutil, 'sensors_battery'):
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
                
        return self.simulated_battery

    def is_charging(self) -> bool:
        """Check if plugged in."""
        if PSUTIL_AVAILABLE and hasattr(psutil, 'sensors_battery'):
            battery = psutil.sensors_battery()
            if battery:
                return battery.power_plugged
        return True # Assume plugged in if unknown (safe default)

    def get_availability_score(self) -> float:
        """
        Calculate 0-100 score for device availability.
        """
        battery = self.estimate_battery_level()
        cpu = self.get_cpu_load()
        mem_pressure = self.check_memory_pressure()
        charging = self.is_charging()
        
        # Base on battery
        score = battery
        
        # Bonus for charging
        if charging:
            score = max(score, 90.0) # If charging, availability is high regardless of battery
            
        # Penalties for load
        if mem_pressure == 'high':
            score *= 0.5
        elif mem_pressure == 'medium':
            score *= 0.8
            
        if cpu > 80:
            score *= 0.6
            
        return score


class AdaptiveStrategy:
    """
    Decides training parameters based on Network and Device status.
    """
    def __init__(self, net_monitor: NetworkMonitor, device_monitor: DeviceAvailability):
        self.net_monitor = net_monitor
        self.device_monitor = device_monitor
        
        # Thresholds from config or default
        self.min_network_score = getattr(config, 'MIN_NETWORK_SCORE', 30.0)
        self.min_availability = getattr(config, 'MIN_AVAILABILITY_SCORE', 20.0)

    def should_participate(self) -> bool:
        """Decide if client should participate in current round."""
        net_score = self.net_monitor.get_network_quality_score()
        dev_score = self.device_monitor.get_availability_score()
        
        if net_score < self.min_network_score:
            return False
            
        if dev_score < self.min_availability:
            return False
            
        return True

    def compute_adaptive_params(self, base_batch_size: int, base_epochs: int) -> AdaptiveParams:
        """
        Compute optimal training parameters.
        """
        net_score = self.net_monitor.get_network_quality_score()
        dev_score = self.device_monitor.get_availability_score()
        
        # 1. Adapt Batch Size (Memory constraint mainly, but also stability)
        batch_size = base_batch_size
        mem_pressure = self.device_monitor.check_memory_pressure()
        
        if mem_pressure == 'high':
            batch_size = max(1, batch_size // 4)
        elif mem_pressure == 'medium':
            batch_size = max(1, batch_size // 2)
            
        # 2. Adapt Local Epochs (Device speed & Battery)
        # If device is struggling, do less work
        local_epochs = base_epochs
        if dev_score < 50:
            local_epochs = max(1, int(base_epochs * 0.5))
        if dev_score < 30:
            local_epochs = max(1, int(base_epochs * 0.2))
            
        # 3. Adapt Compression Level (Network Bandwidth)
        # Low network score -> Heavy compression
        compression_level = 'none' # default
        if net_score < 40:
            compression_level = 'heavy' # e.g., Quantization + Pruning + Sparse update
        elif net_score < 70:
            compression_level = 'light' # e.g., Quantization only
            
        # 4. Upload Strategy
        upload_strategy = {
            'timeout': 60.0,
            'retries': 3
        }
        if net_score < 50:
            upload_strategy['timeout'] = 120.0
            upload_strategy['retries'] = 5
            
        return AdaptiveParams(
            should_participate=True, # Assumed if we called this
            compression_level=compression_level,
            local_epochs=local_epochs,
            batch_size=batch_size,
            upload_strategy=upload_strategy
        )
