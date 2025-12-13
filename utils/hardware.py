"""
Hardware-aware Optimization Module.

This module provides functionalities for:
1. Device Profiling: Analyzing hardware capabilities (CPU, GPU, Memory).
2. Resource Scheduling: Adaptive batch sizes and local epochs based on hardware.
3. Heterogeneous Model Management: Assigning appropriate model complexities to devices.
"""

import time
import torch
import torch.nn as nn
import numpy as np
import os
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import copy

@dataclass
class DeviceProfile:
    """Dataclass to store device hardware capabilities."""
    device_id: str
    device_type: str  # 'cpu', 'cuda', 'edge_low', 'edge_mid', 'edge_high'
    memory_total_mb: float
    memory_available_mb: float
    cpu_count: int
    compute_score: float  # 0-100 score relative to a powerful baseline
    bandwidth_mbps: float = 10.0  # Default value, hard to measure accurately without network test
    power_budget_watts: float = 0.0 # 0.0 means unlimited/unknown
    network_quality_score: float = 100.0  # Network quality (0-100)
    latency_ms: float = 50.0              # Latency to server
    is_network_stable: bool = True        # Is network jitter low
    availability_score: float = 100.0     # Device availability (0-100)
    inference_latency_ms: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        return (f"DeviceProfile(id={self.device_id}, type={self.device_type}, "
                f"mem={self.memory_available_mb:.1f}/{self.memory_total_mb:.1f}MB, "
                f"compute={self.compute_score:.1f}, net={self.network_quality_score:.1f}, "
                f"avail={self.availability_score:.1f})")

class DeviceProfiler:
    """Tools to profile the current device's hardware capabilities."""
    
    def __init__(self, device_id: str = "client_default"):
        self.device_id = device_id
        
    def detect_device(self) -> DeviceProfile:
        """
        Auto-detect system hardware and return a DeviceProfile.
        """
        # Memory Info
        vm = psutil.virtual_memory()
        mem_total = vm.total / (1024**2)
        mem_avail = vm.available / (1024**2)
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        
        # Device Type & Compute Score (Heuristic)
        device_type = 'cpu'
        compute_score = 10.0 # Baseline for weak CPU
        
        if torch.cuda.is_available():
            device_type = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            compute_score = 50.0 # Baseline for GPU
            # Simple heuristic boost for known powerful GPUs
            if '3090' in gpu_name or '4090' in gpu_name or 'A100' in gpu_name:
                compute_score = 90.0
            elif '3080' in gpu_name or '4080' in gpu_name:
                 compute_score = 80.0
            
            # Add VRAM to memory info if using GPU (approximation)
            try:
                # This is just VRAM, strictly speaking we should track system RAM too
                # but for ML, VRAM is often the bottleneck.
                # Here we stick to system RAM as primary generic metric, 
                # but boost compute_score for GPU presence.
                pass 
            except:
                pass
        else:
            # Check for powerful CPUs
            if cpu_count > 16:
                compute_score = 30.0
            elif cpu_count > 8:
                compute_score = 20.0
                
        return DeviceProfile(
            device_id=self.device_id,
            device_type=device_type,
            memory_total_mb=mem_total,
            memory_available_mb=mem_avail,
            cpu_count=cpu_count,
            compute_score=compute_score
        )

    def benchmark_inference(self, model: nn.Module, input_shape: Tuple[int, ...], 
                          num_runs: int = 20, warmup: int = 5) -> float:
        """
        Benchmark model inference latency.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (e.g. (1, 3, 224, 224))
            num_runs: Number of timed runs
            warmup: Number of warmup runs
            
        Returns:
            Average latency in milliseconds
        """
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(dummy_input)
                
            # Timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
                
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
        avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
        return avg_latency_ms

    def benchmark_training(self, model: nn.Module, input_shape: Tuple[int, ...], 
                         batch_size: int = 32, num_steps: int = 10) -> float:
        """
        Benchmark training throughput (samples/second).
        """
        device = next(model.parameters()).device
        # Ensure batch size matches
        shape = list(input_shape)
        shape[0] = batch_size
        dummy_input = torch.randn(shape).to(device)
        dummy_label = torch.randint(0, 10, (batch_size,)).to(device) # Assume classification
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_label)
            loss.backward()
            optimizer.step()
            
        # Timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_label)
            loss.backward()
            optimizer.step()
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_samples = batch_size * num_steps
        throughput_samples_per_sec = total_samples / (end_time - start_time)
        return throughput_samples_per_sec


class ResourceScheduler:
    """
    Decides training configurations based on device profiles.
    """
    
    def compute_optimal_batch_size(self, profile: DeviceProfile, model_size_mb: float, 
                                 sample_size_mb: float = 0.5) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            profile: Device profile
            model_size_mb: Size of model weights
            sample_size_mb: Estimated size of one sample (features + gradients)
            
        Returns:
            Recommended batch size (power of 2)
        """
        # Heuristic: reserve 50% memory for system and safety buffer
        available_mem = profile.memory_available_mb * 0.5
        
        # Memory per batch = Model Size (fwd+bwd copy) + (Batch Size * Sample Size * Overhead)
        # Roughly: overhead factor ~ 3x for optimizer states + activations
        
        remaining_mem = available_mem - (model_size_mb * 2) 
        
        if remaining_mem <= 0:
            return 1 # Minimal
        
        # Calculate max possible batch size
        max_batch = remaining_mem / (sample_size_mb * 4) # 4 is safety factor for activations
        
        # Snap to power of 2
        batch_size = 1
        while batch_size * 2 <= max_batch:
            batch_size *= 2
            
        # Cap at reasonable limits
        if batch_size > 128: batch_size = 128
        if batch_size < 1: batch_size = 1
        
        return int(batch_size)

    def compute_adaptive_epochs(self, profile: DeviceProfile, 
                              latency_budget_ms: float = 5000.0,
                              ms_per_step: float = 100.0) -> int:
        """
        Calculate how many local epochs fit within a latency budget.
        
        Args:
            profile: Device profile
            latency_budget_ms: Total time budget for local training round
            ms_per_step: Measured or estimated time per training step
        """
        # Simple Logic: Latency Budget / Time Per Epoch
        # We need steps per epoch, which depends on dataset size (unknown here)
        # So we return TOTAL STEPS allowed, client can convert to epochs.
        
        # If compute score is low, reduce budget to avoid straggling effect?
        # OR: Straggler mitigation means we assign FEWER steps to slow devices 
        # so they finish at same time as fast devices.
        
        # Let's assume latency_budget is global deadline.
        # Max steps = Budget / ms_per_step
        
        max_steps = int(latency_budget_ms / ms_per_step)
        if max_steps < 1: max_steps = 1
        return max_steps

    def should_participate(self, profile: DeviceProfile, min_battery_level: float = 0.2) -> bool:
        """
        Check if device should participate in this round (e.g. check battery).
        """
        # Simulation: randomly drop node if "battery" (not real) is low or load is high
        # Since we don't have real battery reading, we use a placeholder logic
        # or rely on memory pressure.
        if profile.memory_available_mb < 200: # Less than 200MB free
            return False
        return True


class ModelRegistry:
    """
    Manages different versions of models (e.g. Full, Small, Nano) for heterogeneous devices.
    """
    def __init__(self):
        self.registry = {} # { 'model_name': { 'complexity': int, 'factory': callable } }

    def register_model(self, name: str, complexity_score: int, model_factory):
        """
        Register a model variant.
        Complexity score: higher is more complex (1-100).
        """
        self.registry[name] = {
            'complexity': complexity_score,
            'factory': model_factory
        }

    def get_model(self, name: str):
        return self.registry[name]['factory']()

    def get_best_model_for_device(self, profile: DeviceProfile) -> str:
        """
        Select the best model variant for a device profile.
        """
        # Simple mapping based on compute score
        # Score < 20: Low complexity
        # Score 20-60: Mid complexity
        # Score > 60: High complexity
        
        score = profile.compute_score
        target_complexity = 0
        
        if score > 60:
            target_complexity = 80 # Aim for high
        elif score > 20:
            target_complexity = 50 # Aim for mid
        else:
            target_complexity = 10 # Aim for low
            
        # Find closest match
        best_name = None
        min_diff = float('inf')
        
        for name, info in self.registry.items():
            diff = abs(info['complexity'] - target_complexity)
            if diff < min_diff:
                min_diff = diff
                best_name = name
                
        return best_name

    def get_model_config_for_profile(self, profile: DeviceProfile) -> dict:
        """
        Get model configuration (hyperparameters) suitable for a device profile.
        Specifically tuned for YOLOv11 scalable widths.
        """
        score = profile.compute_score
        
        # Mapping compute score to width_mult
        # > 80: Full capacity (Server/High-end PC) -> 1.0
        # 50-80: High Performance (Gaming Laptop/Mid-PC) -> 0.75
        # 20-50: Mid Performance (Laptop/Edge AI) -> 0.50
        # < 20: Low Power (Mobile/IoT) -> 0.25
        
        if score >= 80:
            width_mult = 1.0
            desc = "High Performance (Full)"
        elif score >= 50:
            width_mult = 0.75
            desc = "Balanced (High)"
        elif score >= 20:
            width_mult = 0.50
            desc = "Edge Optimized (Mid)"
        else:
            width_mult = 0.25
            desc = "Ultra Lightweight (Low)"
            
        return {
            'width_mult': width_mult,
            'description': desc,
            'target_device': profile.device_type
        }

class HeterogeneousManager:
    """
    Server-side manager for heterogeneous clients.
    """
    def __init__(self):
        self.client_profiles: Dict[str, DeviceProfile] = {}
        self.model_registry = ModelRegistry()

    def register_client(self, client_id: str, profile_dict: dict):
        """
        Register or update a client's profile.
        """
        # Parse dict back to DeviceProfile object
        try:
            profile = DeviceProfile(
                device_id=client_id,
                device_type=profile_dict.get('device_type', 'cpu'),
                memory_total_mb=profile_dict.get('memory_total_mb', 0),
                memory_available_mb=profile_dict.get('memory_available_mb', 0),
                cpu_count=profile_dict.get('cpu_count', 1),
                compute_score=profile_dict.get('compute_score', 10.0)
            )
            self.client_profiles[client_id] = profile
        except Exception as e:
            print(f"Error registering client {client_id}: {e}")

    def assign_model(self, client_id: str) -> str:
        """
        Decide which model variant to send to a client.
        """
        profile = self.client_profiles.get(client_id)
        if not profile:
            # Default to simplest model if unknown
            # Or assume standard if we trust unknown clients
            # Let's return a default key if registry has it, else None
            return list(self.registry.registry.keys())[0] if self.registry.registry else None
            
        return self.model_registry.get_best_model_for_device(profile)

