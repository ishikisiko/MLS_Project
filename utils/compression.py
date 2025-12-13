import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import quantize_dynamic
import copy
import os
from utils import config
from utils.detection_loss import DetectionLoss

def quantize_model(model, dtype=torch.qint8):
    """
    Quantize model using dynamic quantization.
    """
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN}, dtype=dtype
    )
    return quantized_model

import io

def get_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes and megabytes.
    Uses serialization for accurate measurement of quantized models.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    total_size = buffer.tell()
    
    # Also count parameters for reference (might be 0 for quantized)
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'param_size_bytes': total_size, # Approximate
        'buffer_size_bytes': 0,
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'num_parameters': num_params
    }

def compare_models(original_model, compressed_model):
    """
    Compare original and compressed model sizes.
    """
    original_info = get_model_size(original_model)
    compressed_info = get_model_size(compressed_model)
    
    # print(f"Original size: {original_info['total_size_mb']:.2f} MB")
    # print(f"Compressed size: {compressed_info['total_size_mb']:.2f} MB")
    
    return compressed_info

import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model):
        self.model = model
    
    def unstructured_prune(self, amount=0.5):
        # Prune all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
        return self.model
    
    def structured_prune(self, amount=0.5, n=2, dim=0):
        """
        Apply structured pruning to the model.
        Prunes entire channels/filters based on L-n norm.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip the final detection head output layers to preserve output shape
                if 'cv3' in name or 'head' in name:
                     continue
                     
                try:
                    prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
                except Exception as e:
                    print(f"Skipping pruning for {name}: {e}")
        return self.model
    
    def get_sparsity(self):
        total_params = 0
        zero_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Check if pruning has been applied (weight_mask exists)
                if hasattr(module, 'weight_mask'):
                    weight = module.weight * module.weight_mask
                else:
                    weight = module.weight
                
                total_params += weight.nelement()
                zero_params += torch.sum(weight == 0).item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return {
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params
        }

    def remove_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        return self.model

class ModelQuantizer:
    def __init__(self, model):
        self.model = model
    
    def dynamic_quantize(self, dtype=torch.qint8):
        return quantize_dynamic(
            self.model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN}, dtype=dtype
        )

    def static_quantize(self, calibration_loader, num_calibration_batches=10):
        """
        Apply static quantization to the model (Post Training Static Quantization).
        Suitable for CNNs (Conv2d layers).
        """
        self.model.eval()
        # Quantization is typically done on CPU
        self.model.to('cpu')
        
        # Wrap model to ensure input is quantized and output is dequantized
        # This is necessary for models that don't have QuantStub/DeQuantStub
        self.model = torch.quantization.QuantWrapper(self.model)
        
        # 1. Set qconfig
        # 'fbgemm' for x86, 'qnnpack' for ARM
        backend = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 2. Prepare (inserts observers)
        torch.quantization.prepare(self.model, inplace=True)
        
        # 3. Calibrate
        print(f"Calibrating model ({backend}) with {num_calibration_batches} batches...")
        with torch.no_grad():
            for i, batch_data in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                
                # Handle different data loader formats
                if isinstance(batch_data, (tuple, list)):
                    if len(batch_data) >= 1:
                        inputs = batch_data[0]
                    else:
                        continue
                else:
                    inputs = batch_data
                
                # Ensure inputs are on CPU
                inputs = inputs.to('cpu')
                self.model(inputs)
        
        # 4. Convert (replaces layers with quantized versions)
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model

class DistillationTrainer:
    """
    Knowledge distillation trainer (Classification).
    """
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def distillation_loss(self, student_logits, teacher_logits, labels):
        T = self.temperature
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T * T)
        hard_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
    def train_step(self, inputs, labels, optimizer, device='cpu'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        self.teacher.to(device)
        self.student.to(device)
        self.student.train()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        student_logits = self.student(inputs)
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

class DetectionDistillationTrainer:
    """
    Knowledge distillation trainer specifically for YOLO object detection models.
    """
    def __init__(self, teacher_model, student_model, alpha=0.5):
        """
        Initialize the detection distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher detection model
            student_model: Student detection model
            alpha: Weight for soft distillation loss (1-alpha for hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        
        # Teacher should be in eval mode and not trainable
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Hard loss function (Standard Detection Loss)
        self.detection_loss = DetectionLoss(num_classes=config.NUM_CLASSES)
        
    def distillation_loss(self, student_pred, teacher_pred):
        """
        Calculate distillation loss between student and teacher outputs.
        
        Args:
            student_pred: (B, N, 4+1+C) student outputs (logits)
            teacher_pred: (B, N, 4+1+C) teacher outputs (logits)
        """
        # Split outputs
        s_box, s_obj, s_cls = student_pred[..., :4], student_pred[..., 4], student_pred[..., 5:]
        t_box, t_obj, t_cls = teacher_pred[..., :4], teacher_pred[..., 4], teacher_pred[..., 5:]
        
        # 1. Objectness Loss
        # Use teacher's sigmoid probability as soft target for student
        t_obj_prob = torch.sigmoid(t_obj)
        # BCEWithLogits accepts logits as input and probabilities as target
        obj_loss = F.binary_cross_entropy_with_logits(s_obj, t_obj_prob)
        
        # 2. Classification Loss
        # Distill class probabilities
        t_cls_prob = torch.sigmoid(t_cls)
        cls_loss = F.binary_cross_entropy_with_logits(s_cls, t_cls_prob)
        
        # 3. Box Loss
        # Only distill box predictions where teacher is confident (obj > 0.5)
        # Using MSE on sigmoid-normalized box coordinates
        mask = t_obj_prob > 0.5
        
        if mask.sum() > 0:
            s_box_sig = torch.sigmoid(s_box[mask])
            t_box_sig = torch.sigmoid(t_box[mask])
            box_loss = F.mse_loss(s_box_sig, t_box_sig)
        else:
            box_loss = torch.tensor(0.0, device=student_pred.device)
            
        # Weighted sum (using same weights as config)
        total_loss = (
            config.BOX_WEIGHT * box_loss + 
            config.OBJ_WEIGHT * obj_loss + 
            config.CLS_WEIGHT * cls_loss
        )
        
        return total_loss

    def train_step(self, inputs, targets, optimizer, device='cpu'):
        """
        Perform one training step of detection knowledge distillation.
        
        Args:
            inputs: Images (B, 3, H, W)
            targets: Ground truth [batch_idx, cls, x, y, w, h]
        """
        inputs = inputs.to(device)
        targets = targets.to(device)
        self.teacher.to(device)
        self.student.to(device)
        
        self.student.train()
        optimizer.zero_grad()
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_pred = self.teacher(inputs)
        
        # Get student predictions
        student_pred = self.student(inputs)
        
        # Calculate losses
        # 1. Hard target loss (Ground Truth)
        hard_loss, _ = self.detection_loss(student_pred, targets)
        
        # 2. Soft target loss (Distillation)
        soft_loss = self.distillation_loss(student_pred, teacher_pred)
        
        # Combine
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
