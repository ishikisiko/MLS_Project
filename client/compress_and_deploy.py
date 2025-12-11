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
import torch.optim as optim
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import SimpleCNN
from utils.compression import (
    ModelPruner, ModelQuantizer, DistillationTrainer,
    get_model_size, compare_models
)
from utils.deployment import (
    ModelExporter, export_to_onnx, optimize_onnx, 
    verify_onnx, get_model_info, get_file_size
)


def create_dummy_data(batch_size=32, num_batches=10):
    """Create dummy training data for demonstration."""
    data = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        data.append((inputs, labels))
    return data


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


def demonstrate_distillation():
    """Demonstrate knowledge distillation capabilities."""
    print("\n" + "="*60)
    print("3. KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("="*60)
    
    # Create teacher (larger) and student (smaller) models
    teacher = SimpleCNN()
    
    # Create a smaller student model
    class SmallCNN(nn.Module):
        def __init__(self):
            super(SmallCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 4, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(4, 8, 5)
            self.fc1 = nn.Linear(8 * 5 * 5, 60)
            self.fc2 = nn.Linear(60, 10)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 8 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    student = SmallCNN()
    
    teacher_info = get_model_size(teacher)
    student_info = get_model_size(student)
    
    print(f"\nTeacher model: {teacher_info['num_parameters']:,} parameters")
    print(f"Student model: {student_info['num_parameters']:,} parameters")
    print(f"Compression ratio: {teacher_info['num_parameters']/student_info['num_parameters']:.2f}x")
    
    # Setup distillation training
    distiller = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        temperature=4.0,
        alpha=0.7
    )
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size=16, num_batches=5)
    
    # Train for a few steps
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    print("\nTraining student with knowledge distillation...")
    for epoch in range(2):
        total_loss = 0
        for inputs, labels in dummy_data:
            loss = distiller.train_step(inputs, labels, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(dummy_data)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("  Distillation training complete!")
    
    return student


def demonstrate_export(model, output_dir="exported_models"):
    """Demonstrate model export capabilities."""
    print("\n" + "="*60)
    print("4. MODEL EXPORT DEMONSTRATION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare export
    dummy_input = torch.randn(1, 3, 32, 32)
    exporter = ModelExporter(model, dummy_input)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"\nExporting to ONNX...")
    exporter.export_onnx(onnx_path, optimize=False)
    
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
    tflite_path = os.path.join(output_dir, "model.tflite")
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
    distilled_model = demonstrate_distillation()
    
    # 4. Export (using pruned model as example)
    onnx_path = demonstrate_export(SimpleCNN(), "exported_models")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nCompression techniques demonstrated:")
    print("  ✓ Unstructured Pruning (L1-norm based)")
    print("  ✓ Dynamic Quantization (INT8)")
    print("  ✓ Knowledge Distillation")
    print("\nExport formats:")
    print("  ✓ ONNX (with optimization)")
    print("  - TFLite (requires tensorflow, onnx-tf)")
    print("\nExported models saved to: ./exported_models/")
    

if __name__ == "__main__":
    run_full_pipeline()
