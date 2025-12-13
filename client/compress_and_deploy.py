"""
Model Compression and Deployment Example Script.

Demonstrates the complete workflow of:
1. Training/loading a model
2. Applying compression techniques (pruning, quantization)
3. Exporting to deployment formats (ONNX, TFLite)
4. Comparing model sizes and verifying outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import SimpleCNN
from utils.detection_models import YOLOv11n
from utils.compression import (
    ModelPruner, ModelQuantizer, DistillationTrainer, DetectionDistillationTrainer,
    get_model_size, compare_models
)
from utils.deployment import (
    ModelExporter, export_to_onnx, optimize_onnx, 
    verify_onnx, get_model_info, get_file_size
)
from utils.data_loader import get_data_loaders
from utils import config


from utils.detection_loss import DetectionLoss
try:
    from client.training import evaluate_detection
except ImportError:
    from training import evaluate_detection

def demonstrate_pruning(model, val_loader=None):
    """Demonstrate model pruning capabilities."""
    print("\n" + "="*60)
    print("1. MODEL PRUNING DEMONSTRATION (Structured)")
    print("="*60)
    
    # Get original model info
    original_info = get_model_size(model)
    print(f"\nOriginal model:")
    print(f"  - Parameters: {original_info['num_parameters']:,}")
    print(f"  - Size: {original_info['total_size_mb']:.4f} MB")
    
    # Evaluate original mAP
    if val_loader:
        print("  Evaluating original model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metrics = evaluate_detection(model, val_loader, device, num_classes=config.NUM_CLASSES)
        print(f"  Original mAP@0.5: {metrics['mAP@0.5']:.4f}")

    # Apply structured pruning
    pruner = ModelPruner(model)
    # Use structured pruning now
    pruned_model = pruner.structured_prune(amount=0.5)
    
    # Check sparsity
    sparsity_info = pruner.get_sparsity()
    print(f"\nAfter 50% structured pruning:")
    print(f"  - Sparsity: {sparsity_info['sparsity']*100:.2f}%")
    print(f"  - Zero weights: {sparsity_info['zero_params']:,} / {sparsity_info['total_params']:,}")
    
    # Evaluate pruned mAP
    if val_loader:
        print("  Evaluating pruned model...")
        model.to(device)
        metrics = evaluate_detection(model, val_loader, device, num_classes=config.NUM_CLASSES)
        print(f"  Pruned mAP@0.5: {metrics['mAP@0.5']:.4f}")

    # Make pruning permanent
    pruner.remove_pruning()
    print("  - Pruning masks removed (weights now permanent zeros)")
    
    return pruned_model


def demonstrate_quantization(model, calibration_loader, val_loader=None):
    """Demonstrate model quantization capabilities."""
    print("\n" + "="*60)
    print("2. MODEL QUANTIZATION DEMONSTRATION")
    print("="*60)
    
    # Get original size
    original_info = get_model_size(model)
    print(f"\nOriginal model size: {original_info['total_size_mb']:.4f} MB")

    # Evaluate original mAP
    if val_loader:
        print("  Evaluating original model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metrics = evaluate_detection(model, val_loader, device, num_classes=config.NUM_CLASSES)
        print(f"  Original mAP@0.5: {metrics['mAP@0.5']:.4f}")
    
    # Static quantization
    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.static_quantize(calibration_loader)
    
    quantized_info = get_model_size(quantized_model)
    print(f"\nAfter static INT8 quantization:")
    print(f"  - Size: {quantized_info['total_size_mb']:.4f} MB")
    
    # Compare
    if original_info['total_size_bytes'] > 0:
        reduction = (1 - quantized_info['total_size_bytes'] / original_info['total_size_bytes']) * 100
        print(f"  - Size reduction: {reduction:.2f}%")

    # Evaluate quantized mAP
    if val_loader:
        print("  Evaluating quantized model...")
        # Quantized model runs on CPU usually
        metrics = evaluate_detection(quantized_model, val_loader, device='cpu', num_classes=config.NUM_CLASSES)
        print(f"  Quantized mAP@0.5: {metrics['mAP@0.5']:.4f}")
    
    return quantized_model


def demonstrate_distillation(train_loader, val_loader=None, teacher_model=None):
    """Demonstrate knowledge distillation capabilities."""
    print("\n" + "="*60)
    print("3. KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("="*60)
    
    # Create teacher (larger) and student (smaller) models
    # Teacher: 1.0x width (Standard YOLOv11n Baseline)
    if teacher_model is not None:
        print("Using provided Teacher Model (Trained Baseline)...")
        teacher = teacher_model
    else:
        print("Initializing Teacher Model (YOLOv11n width=1.0 - Baseline)...")
        teacher = YOLOv11n(width_mult=1.0)
    
    # Student: 0.5x width (Compressed)
    print("Initializing Student Model (YOLOv11n width=0.5 - Compressed)...")
    student = YOLOv11n(width_mult=0.5)
    
    teacher_info = get_model_size(teacher)
    student_info = get_model_size(student)
    
    print(f"\nTeacher model: {teacher_info['num_parameters']:,} parameters")
    print(f"Student model: {student_info['num_parameters']:,} parameters")
    print(f"Compression ratio: {teacher_info['num_parameters']/student_info['num_parameters']:.2f}x")
    
    # Setup distillation training
    distiller = DetectionDistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        alpha=0.5
    )
    
    # Train for a few steps
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    print("\nTraining student with knowledge distillation using real dataset (UA-DETRAC)...")
    # Using a subset of batches for demonstration purposes
    max_batches = 50 
    
    for epoch in range(10): # Changed to 10 epochs
        total_loss = 0
        batch_count = 0
        
        # Handle the detection dataloader format (images, targets, paths, shapes)
        for batch_data in train_loader:
            if len(batch_data) == 4:
                inputs, targets, _, _ = batch_data
            else:
                continue

            # DetectionDistillationTrainer handles device movement internally or we can do it here
            # Ideally trainer handles it.
            
            # Loss computation
            loss = distiller.train_step(inputs, targets, optimizer)
            total_loss += loss
            batch_count += 1
            if batch_count >= max_batches:
                break
                
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f} (over {batch_count} batches)")
            
            # Evaluate mAP if val_loader is provided
            if val_loader is not None:
                print("  Evaluating student model on validation set...")
                # Assuming cuda if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                student.to(device)
                metrics = evaluate_detection(student, val_loader, device, num_classes=4)
                print(f"  Student mAP@0.5: {metrics['mAP@0.5']:.4f}")
                print(f"  Student mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    
    print("  Distillation training complete!")
    
    return student


def demonstrate_export(model, output_dir="exported_models", filename="model.onnx", input_shape=(1, 3, 32, 32)):
    """Demonstrate model export capabilities."""
    print("\n" + "="*60)
    print(f"4. MODEL EXPORT DEMONSTRATION: {filename}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare export
    dummy_input = torch.randn(*input_shape)
    exporter = ModelExporter(model, dummy_input)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, filename)
    print(f"\nExporting to ONNX ({filename})...")
    try:
        exporter.export_onnx(onnx_path, optimize=False)
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return None
    
    # Check file size
    onnx_size = get_file_size(onnx_path)
    if onnx_size:
        print(f"  ONNX file size: {onnx_size['size_mb']:.4f} MB")
    
    # Verify ONNX export
    print("\nVerifying ONNX export...")
    exporter.verify(onnx_path)
    
    # Try ONNX optimization
    print("\nOptimizing ONNX model...")
    try:
        optimized_path = optimize_onnx(onnx_path)
        opt_size = get_file_size(optimized_path)
        if opt_size:
            print(f"  Optimized ONNX size: {opt_size['size_mb']:.4f} MB")
    except Exception as e:
        print(f"  Optimization skipped: {e}")
    
    # Try TFLite export (may fail if dependencies not installed)
    print("\nAttempting TFLite export...")
    tflite_filename = os.path.splitext(filename)[0] + ".tflite"
    tflite_path = os.path.join(output_dir, tflite_filename)
    try:
        from utils.deployment import export_to_tflite
        result = export_to_tflite(onnx_path, tflite_path)
        if result:
            tflite_size = get_file_size(tflite_path)
            if tflite_size:
                print(f"  TFLite file size: {tflite_size['size_mb']:.4f} MB")
        else:
            print("  TFLite export not available (missing dependencies)")
    except Exception as e:
        print(f"  TFLite export skipped: {e}")
    
    return onnx_path


def demonstrate_cross_platform_distillation(train_loader, teacher_model=None):
    """Demonstrate distillation adapted for different target devices."""
    print("\n" + "="*60)
    print("5. CROSS-PLATFORM KNOWLEDGE TRANSFER")
    print("="*60)
    
    # 1. Define Simulated Device Profiles
    from utils.hardware import DeviceProfile, ModelRegistry
    
    profiles = [
        DeviceProfile("server_gpu", "cuda", 16000, 12000, 16, 90.0), # High-end
        DeviceProfile("edge_mbp", "cpu", 8000, 4000, 8, 40.0),       # Mid-range
        DeviceProfile("iot_cam", "edge_low", 1024, 256, 4, 10.0)     # Low-end
    ]
    
    registry = ModelRegistry()
    
    # Teacher is constant
    if teacher_model is not None:
        print("Using provided Teacher Model (Trained Baseline)...")
        teacher = teacher_model
    else:
        print("Initializing Teacher Model (YOLOv11n width=1.0)...")
        teacher = YOLOv11n(width_mult=1.0)
    
    for profile in profiles:
        print(f"\n{'-'*40}")
        print(f"Targeting Device: {profile.device_id} (Score: {profile.compute_score})")
        
        # 2. Get optimal config
        config_map = registry.get_model_config_for_profile(profile)
        width = config_map['width_mult']
        desc = config_map['description']
        
        print(f"Selected Architecture: YOLOv11n width={width} ({desc})")
        
        # 3. Instantiate Student
        student = YOLOv11n(width_mult=width)
        student_info = get_model_size(student)
        print(f"Student Parameters: {student_info['num_parameters']:,}")
        
        # 4. Brief Distillation (Simulation)
        distiller = DetectionDistillationTrainer(teacher, student, alpha=0.5)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        
        print("Transferring knowledge...")
        # Train 1 batch just to verify flow
        for batch_data in train_loader:
             if len(batch_data) == 4:
                inputs, targets, _, _ = batch_data
                loss = distiller.train_step(inputs, targets, optimizer)
                print(f"  Batch Loss: {loss:.4f}")
                break # Just 1 batch for demo
        
        print(f"  -> Model ready for {profile.device_id}")


def run_full_pipeline():
    """Run the complete compression and deployment pipeline."""
    print("\n" + "="*60)
    print("MODEL COMPRESSION & DEPLOYMENT PIPELINE")
    print("="*60)
    
    # Load Real Data
    print(f"\nLoading UA-DETRAC dataset (resized to {config.INPUT_SIZE}x{config.INPUT_SIZE})...")
    
    # Check if data exists
    if not os.path.exists(config.DEFAULT_DATA_ROOT) or not os.path.exists(os.path.join(config.DEFAULT_DATA_ROOT, 'train')):
        print(f"Dataset not found at {config.DEFAULT_DATA_ROOT}. Switching to MOCK data for demonstration.")
        config.USE_MOCK_DATA = True

    train_loader, test_loader = get_data_loaders(batch_size=32)
    print("Data loaded successfully.")

    # Create model
    print("\nCreating YOLOv11n model...")
    model = YOLOv11n()
    
    # Load baseline model if it exists
    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_model.pth")
    if os.path.exists(baseline_path):
        print(f"Loading trained baseline model from {baseline_path}...")
        try:
            state_dict = torch.load(baseline_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random initialization instead.")
    else:
        print(f"Warning: {baseline_path} not found. Using random initialization.")

    model_info = get_model_info(model)
    print(f"Model created: {model_info['num_parameters']:,} parameters, "
          f"{model_info['size_mb']:.4f} MB")
    
    # Keep a copy of the state dict to reset for each demonstration
    original_state_dict = model.state_dict()
    
    # 1. Pruning
    # Create a fresh instance with loaded weights
    pruning_model = YOLOv11n()
    pruning_model.load_state_dict(original_state_dict)
    pruned_model = demonstrate_pruning(pruning_model, val_loader=test_loader)
    
    # 2. Quantization (using loaded model)
    quant_model = YOLOv11n()
    quant_model.load_state_dict(original_state_dict)
    quantized_model = demonstrate_quantization(quant_model, train_loader, val_loader=test_loader)
    
    # 3. Distillation (Standard)
    # Use loaded model as teacher
    teacher_model = YOLOv11n()
    teacher_model.load_state_dict(original_state_dict)
    distilled_model = demonstrate_distillation(train_loader, test_loader, teacher_model=teacher_model)
    
    # 4. Cross-Platform Distillation (New)
    # Use loaded model as teacher
    cp_teacher_model = YOLOv11n()
    cp_teacher_model.load_state_dict(original_state_dict)
    demonstrate_cross_platform_distillation(train_loader, teacher_model=cp_teacher_model)
    
    # 5. Export all models
    print("\n" + "="*60)
    print("EXPORTING ALL MODELS")
    print("="*60)
    
    demonstrate_export(pruned_model, "exported_models", "pruned_model.onnx", input_shape=(1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    demonstrate_export(quantized_model, "exported_models", "quantized_model.onnx", input_shape=(1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    demonstrate_export(distilled_model, "exported_models", "distilled_model.onnx", input_shape=(1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nCompression techniques demonstrated:")
    print("  ✓ Structured Pruning (L-n norm based)")
    print("  ✓ Static Quantization (INT8)")
    print("  ✓ Knowledge Distillation (UA-DETRAC)")
    print("  ✓ Cross-Platform Architecture Adaptation")
    print("\nExport formats:")
    print("  ✓ ONNX (with optimization)")
    print("  - TFLite (requires tensorflow, onnx-tf)")
    print("\nExported models saved to: ./exported_models/")
    

if __name__ == "__main__":
    run_full_pipeline()

