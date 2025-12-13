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


def demonstrate_pruning(model):
    """Demonstrate model pruning capabilities."""
    print("\n" + "="*60)
    print("1. MODEL PRUNING DEMONSTRATION")
    print("="*60)
    
    # Get original model info
    original_info = get_model_size(model)
    print(f"\nOriginal model:")
    print(f"  - Parameters: {original_info['num_parameters']:,}")
    print(f"  - Size: {original_info['total_size_mb']:.4f} MB")
    
    # Apply unstructured pruning
    pruner = ModelPruner(model)
    pruned_model = pruner.unstructured_prune(amount=0.5)
    
    # Check sparsity
    sparsity_info = pruner.get_sparsity()
    print(f"\nAfter 50% unstructured pruning:")
    print(f"  - Sparsity: {sparsity_info['sparsity']*100:.2f}%")
    print(f"  - Zero weights: {sparsity_info['zero_params']:,} / {sparsity_info['total_params']:,}")
    
    # Make pruning permanent
    pruner.remove_pruning()
    print("  - Pruning masks removed (weights now permanent zeros)")
    
    return pruned_model


def demonstrate_quantization(model):
    """Demonstrate model quantization capabilities."""
    print("\n" + "="*60)
    print("2. MODEL QUANTIZATION DEMONSTRATION")
    print("="*60)
    
    # Get original size
    original_info = get_model_size(model)
    print(f"\nOriginal model size: {original_info['total_size_mb']:.4f} MB")
    
    # Dynamic quantization
    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.dynamic_quantize()
    
    quantized_info = get_model_size(quantized_model)
    print(f"\nAfter dynamic INT8 quantization:")
    print(f"  - Size: {quantized_info['total_size_mb']:.4f} MB")
    
    # Compare
    if original_info['total_size_bytes'] > 0:
        reduction = (1 - quantized_info['total_size_bytes'] / original_info['total_size_bytes']) * 100
        print(f"  - Size reduction: {reduction:.2f}%")
    
    return quantized_model


def demonstrate_distillation(train_loader):
    """Demonstrate knowledge distillation capabilities."""
    print("\n" + "="*60)
    print("3. KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("="*60)
    
    # Create teacher (larger) and student (smaller) models
    # Teacher: 1.0x width (Standard YOLOv11n Baseline)
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
    max_batches = 10 
    
    for epoch in range(1): # Reduced to 1 epoch for demo speed
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


def run_full_pipeline():
    """Run the complete compression and deployment pipeline."""
    print("\n" + "="*60)
    print("MODEL COMPRESSION & DEPLOYMENT PIPELINE")
    print("="*60)
    
    # Load Real Data
    print(f"\nLoading UA-DETRAC dataset (resized to {config.INPUT_SIZE}x{config.INPUT_SIZE})...")
    train_loader, test_loader = get_data_loaders(batch_size=32)
    print("Data loaded successfully.")

    # Create model
    print("\nCreating SimpleCNN model...")
    model = SimpleCNN()
    model_info = get_model_info(model)
    print(f"Model created: {model_info['num_parameters']:,} parameters, "
          f"{model_info['size_mb']:.4f} MB")
    
    # 1. Pruning
    pruned_model = demonstrate_pruning(model)
    
    # 2. Quantization (using fresh model)
    fresh_model = SimpleCNN()
    quantized_model = demonstrate_quantization(fresh_model)
    
    # 3. Distillation
    distilled_model = demonstrate_distillation(train_loader)
    
    # 4. Export all models
    print("\n" + "="*60)
    print("EXPORTING ALL MODELS")
    print("="*60)
    
    demonstrate_export(pruned_model, "exported_models", "pruned_model.onnx")
    demonstrate_export(quantized_model, "exported_models", "quantized_model.onnx")
    demonstrate_export(distilled_model, "exported_models", "distilled_model.onnx", input_shape=(1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nCompression techniques demonstrated:")
    print("  ✓ Unstructured Pruning (L1-norm based)")
    print("  ✓ Dynamic Quantization (INT8)")
    print("  ✓ Knowledge Distillation (UA-DETRAC)")
    print("\nExport formats:")
    print("  ✓ ONNX (with optimization)")
    print("  - TFLite (requires tensorflow, onnx-tf)")
    print("\nExported models saved to: ./exported_models/")
    

if __name__ == "__main__":
    run_full_pipeline()
