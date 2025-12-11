import torch
import numpy as np

class DPEngine:
    """
    Differential Privacy Engine for simulating DP-SGD.
    """
    def __init__(self, max_norm: float, noise_multiplier: float):
        """
        Args:
            max_norm: The maximum norm for gradient clipping.
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to the sensitivity.
        """
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier

    def per_sample_clip(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Simulates per-sample gradient clipping.
        Since we might simulate aggregated gradients, if 'gradients' represents a batch average,
        this acts as a global clip. For true per-sample, input should be per-sample gradients.
        
        For this simulation, we assume 'gradients' is a list of tensors representing the model update.
        We treat the entire update vector as one "sample" update in the context of FedAvg client update.
        """
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in gradients]), 2)
        clip_coef = self.max_norm / (total_norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        
        clipped_grads = [g.detach() * clip_coef for g in gradients]
        return clipped_grads

    def add_noise(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Adds Gaussian noise to the gradients.
        Noise standard deviation = noise_multiplier * max_norm
        """
        sigma = self.noise_multiplier * self.max_norm
        noisy_grads = []
        for g in gradients:
            noise = torch.normal(mean=0.0, std=sigma, size=g.shape, device=g.device)
            noisy_grads.append(g + noise)
        return noisy_grads

    def step(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Applies clipping and noise injection.
        """
        clipped = self.per_sample_clip(gradients)
        noisy = self.add_noise(clipped)
        return noisy
