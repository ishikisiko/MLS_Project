"""
Model Compression Module for Federated Learning Deployment.

Provides pruning, quantization, and knowledge distillation capabilities
to reduce model size and improve inference efficiency for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, prepare, convert
import copy


class ModelPruner:
    """
    模型剪枝器，支持结构化和非结构化剪枝。
    
    Supports:
    - Unstructured pruning: L1-norm based weight pruning
    - Structured pruning: Filter-level pruning for Conv layers
    """
    
    def __init__(self, model):
        """
        Initialize the pruner with a model.
        
        Args:
            model: PyTorch model to prune
        """
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())
    
    def unstructured_prune(self, amount=0.3, prune_bias=False):
        """
        Apply L1 unstructured pruning to all Conv2d and Linear layers.
        
        Args:
            amount: Fraction of weights to prune (0.0 to 1.0)
            prune_bias: Whether to also prune bias terms
            
        Returns:
            The pruned model
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                if prune_bias and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=amount)
        
        return self.model
    
    def structured_prune(self, amount=0.3, dim=0):
        """
        Apply structured pruning (filter-level) to Conv2d layers.
        
        Args:
            amount: Fraction of filters to prune
            dim: Dimension along which to prune (0 for output channels)
            
        Returns:
            The pruned model
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=dim)
        
        return self.model
    
    def get_sparsity(self):
        """
        Calculate the overall sparsity of the model.
        
        Returns:
            dict: Contains total params, zero params, and sparsity ratio
        """
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'sparsity': sparsity
        }
    
    def remove_pruning(self):
        """
        Make pruning permanent by removing the pruning reparametrization.
        After this, the pruned weights become actual zeros in the model.
        
        Returns:
            The model with permanent pruning
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning mask exists
                    
                if hasattr(module, 'bias_orig'):
                    try:
                        prune.remove(module, 'bias')
                    except ValueError:
                        pass
        
        return self.model
    
    def reset(self):
        """Reset model to original unpruned state."""
        self.model.load_state_dict(self.original_state)
        return self.model


class ModelQuantizer:
    """
    模型量化器，支持动态量化、静态量化和量化感知训练。
    
    Supports:
    - Dynamic quantization: Weights quantized, activations quantized at runtime
    - Static quantization: Both weights and activations quantized with calibration
    - QAT: Quantization-aware training for higher accuracy
    """
    
    def __init__(self, model):
        """
        Initialize the quantizer with a model.
        
        Args:
            model: PyTorch model to quantize
        """
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())
    
    def dynamic_quantize(self, dtype=torch.qint8):
        """
        Apply dynamic quantization to Linear and LSTM layers.
        
        Args:
            dtype: Quantization dtype (torch.qint8 or torch.float16)
            
        Returns:
            Dynamically quantized model
        """
        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM},
            dtype=dtype
        )
        return quantized_model
    
    def prepare_static_quantization(self, backend='fbgemm'):
        """
        Prepare model for static quantization by inserting observers.
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            
        Returns:
            Model prepared for calibration
        """
        self.model.eval()
        
        # Set quantization config
        if backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
            self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        else:
            torch.backends.quantized.engine = 'fbgemm'
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse modules if applicable (conv+bn+relu, etc.)
        # This needs to be customized per model architecture
        
        # Prepare for calibration
        prepared_model = prepare(self.model)
        return prepared_model
    
    def convert_static_quantization(self, prepared_model):
        """
        Convert a calibrated model to a quantized model.
        
        Args:
            prepared_model: Model that has been prepared and calibrated
            
        Returns:
            Statically quantized model
        """
        quantized_model = convert(prepared_model)
        return quantized_model
    
    def prepare_qat(self, backend='fbgemm'):
        """
        Prepare model for Quantization-Aware Training.
        
        Args:
            backend: Quantization backend
            
        Returns:
            Model prepared for QAT (can be trained)
        """
        self.model.train()
        
        if backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        else:
            torch.backends.quantized.engine = 'fbgemm'
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        qat_model = torch.quantization.prepare_qat(self.model)
        return qat_model
    
    def convert_qat(self, qat_model):
        """
        Convert a QAT-trained model to a quantized model.
        
        Args:
            qat_model: Model that has been trained with QAT
            
        Returns:
            Quantized model
        """
        qat_model.eval()
        quantized_model = convert(qat_model)
        return quantized_model


class DistillationTrainer:
    """
    知识蒸馏训练器，将教师模型的知识转移到学生模型。
    
    Uses a combination of:
    - Soft targets: KL divergence between teacher and student logits
    - Hard targets: Cross-entropy with ground truth labels
    """
    
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher model (larger)
            student_model: Student model to train (smaller)
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Teacher should be in eval mode and not trainable
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=None):
        """
        Calculate the combined distillation loss.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            labels: Ground truth labels
            temperature: Override default temperature
            
        Returns:
            Combined loss value
        """
        T = temperature if temperature is not None else self.temperature
        
        # Soft targets loss (KL Divergence)
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T * T)
        
        # Hard targets loss (Cross Entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss
    
    def train_step(self, inputs, labels, optimizer, device='cpu'):
        """
        Perform one training step of knowledge distillation.
        
        Args:
            inputs: Input batch
            labels: Ground truth labels
            optimizer: Optimizer for student model
            device: Device to use for computation
            
        Returns:
            Loss value for this step
        """
        inputs = inputs.to(device)
        labels = labels.to(device)
        self.teacher.to(device)
        self.student.to(device)
        
        self.student.train()
        
        optimizer.zero_grad()
        
        # Get teacher predictions (no gradient needed)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Get student predictions
        student_logits = self.student(inputs)
        
        # Calculate distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_loader, optimizer, device='cpu'):
        """
        Train for one epoch using knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for student model
            device: Device to use
            
        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0
        
        for inputs, labels in train_loader:
            loss = self.train_step(inputs, labels, optimizer, device)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def get_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes and megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Contains size in bytes, MB, and parameter count
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'param_size_bytes': param_size,
        'buffer_size_bytes': buffer_size,
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'num_parameters': sum(p.numel() for p in model.parameters())
    }


def compare_models(original_model, compressed_model):
    """
    Compare original and compressed model sizes.
    
    Args:
        original_model: Original PyTorch model
        compressed_model: Compressed/quantized model
        
    Returns:
        dict: Comparison metrics
    """
    orig_info = get_model_size(original_model)
    comp_info = get_model_size(compressed_model)
    
    compression_ratio = orig_info['total_size_bytes'] / comp_info['total_size_bytes'] \
        if comp_info['total_size_bytes'] > 0 else float('inf')
    
    return {
        'original': orig_info,
        'compressed': comp_info,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0
    }
