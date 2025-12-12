"""
Model Deployment Module for Federated Learning.

Provides utilities to export PyTorch models to various deployment formats
including ONNX and TensorFlow Lite for edge deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import warnings


def export_to_onnx(model, dummy_input, output_path, 
                   input_names=None, output_names=None,
                   dynamic_axes=None, opset_version=18,
                   verbose=False):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor with correct shape
        output_path: Path to save the .onnx file
        input_names: List of input tensor names
        output_names: List of output tensor names
        dynamic_axes: Dict specifying dynamic axes for inputs/outputs
        opset_version: ONNX opset version (default 18)
        verbose: Whether to print export details
        
    Returns:
        str: Path to the exported ONNX model
    """
    model.eval()
    
    # Default names if not provided
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Default dynamic axes for batch size
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        verbose=verbose,
        do_constant_folding=True  # Optimize by folding constants
    )
    
    print(f"Model exported to ONNX: {output_path}")
    return output_path


def optimize_onnx(input_path, output_path=None, optimization_passes=None):
    """
    Optimize an ONNX model using onnxoptimizer.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model (defaults to input_path)
        optimization_passes: List of optimization passes to apply
        
    Returns:
        str: Path to optimized model
    """
    try:
        import onnx
        from onnxoptimizer import optimize
    except ImportError:
        warnings.warn("onnx and onnxoptimizer packages required for optimization. "
                     "Install with: pip install onnx onnxoptimizer")
        return input_path
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_optimized{ext}"
    
    # Default optimization passes
    if optimization_passes is None:
        optimization_passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_pad',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
        ]
    
    # Load and optimize
    model = onnx.load(input_path)
    optimized_model = optimize(model, optimization_passes)
    
    # Save optimized model
    onnx.save(optimized_model, output_path)
    print(f"Optimized ONNX model saved to: {output_path}")
    
    return output_path


def verify_onnx(onnx_path, dummy_input, pytorch_model, rtol=1e-3, atol=1e-5):
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        dummy_input: Example input tensor
        pytorch_model: Original PyTorch model
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        bool: True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("onnxruntime required for verification")
        return False
    
    pytorch_model.eval()
    
    # Get PyTorch output
    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            pytorch_output = pytorch_model(*dummy_input).numpy()
            ort_input = {f'input_{i}': inp.numpy() for i, inp in enumerate(dummy_input)}
        else:
            pytorch_output = pytorch_model(dummy_input).numpy()
            ort_input = {'input': dummy_input.numpy()}
    
    # Get ONNX Runtime output
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    ort_output = session.run(None, {input_name: dummy_input.numpy() if not isinstance(dummy_input, tuple) else dummy_input[0].numpy()})[0]
    
    # Compare outputs
    is_close = np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("✓ ONNX model verification passed")
    else:
        max_diff = np.max(np.abs(pytorch_output - ort_output))
        print(f"✗ ONNX model verification failed. Max difference: {max_diff}")
    
    return is_close


def export_to_tflite(onnx_path, output_path=None, quantize=False):
    """
    Convert an ONNX model to TensorFlow Lite format.
    
    This uses the ONNX -> TensorFlow -> TFLite pipeline.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save .tflite file
        quantize: Whether to apply post-training quantization
        
    Returns:
        str: Path to TFLite model, or None if conversion failed
    """
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx
    except ImportError as e:
        warnings.warn(f"Required packages not found: {e}. "
                     "Install with: pip install tensorflow onnx-tf")
        return None
    
    if output_path is None:
        base = os.path.splitext(onnx_path)[0]
        output_path = f"{base}.tflite"
    
    # Temporary directory for TF SavedModel
    temp_tf_path = os.path.splitext(onnx_path)[0] + "_tf_temp"
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(temp_tf_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_tf_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to: {output_path}")
        
        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_tf_path):
            shutil.rmtree(temp_tf_path)
        
        return output_path
        
    except Exception as e:
        warnings.warn(f"TFLite conversion failed: {e}")
        return None


class ModelExporter:
    """
    统一的模型导出器，支持多种部署格式。
    
    Unified model exporter supporting ONNX, TFLite, and other formats.
    """
    
    def __init__(self, model, dummy_input):
        """
        Initialize the exporter.
        
        Args:
            model: PyTorch model to export
            dummy_input: Example input tensor for tracing
        """
        self.model = model
        self.dummy_input = dummy_input
        self.model.eval()
    
    def export_onnx(self, output_path, optimize=False, **kwargs):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path for output .onnx file
            optimize: Whether to run ONNX optimization
            **kwargs: Additional arguments for export_to_onnx
            
        Returns:
            str: Path to exported model
        """
        onnx_path = export_to_onnx(self.model, self.dummy_input, output_path, **kwargs)
        
        if optimize:
            onnx_path = optimize_onnx(onnx_path)
        
        return onnx_path
    
    def export_tflite(self, output_path, quantize=False, onnx_path=None):
        """
        Export model to TensorFlow Lite format.
        
        Args:
            output_path: Path for output .tflite file
            quantize: Apply post-training quantization
            onnx_path: Optional pre-existing ONNX model path
            
        Returns:
            str: Path to TFLite model or None if failed
        """
        # First export to ONNX if not provided
        if onnx_path is None:
            base = os.path.splitext(output_path)[0]
            onnx_path = f"{base}_temp.onnx"
            export_to_onnx(self.model, self.dummy_input, onnx_path)
        
        return export_to_tflite(onnx_path, output_path, quantize=quantize)
    
    def export_all(self, output_dir, base_name="model"):
        """
        Export model to all supported formats.
        
        Args:
            output_dir: Directory to save exported models
            base_name: Base name for exported files
            
        Returns:
            dict: Paths to all exported models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Export ONNX
        onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
        results['onnx'] = self.export_onnx(onnx_path, optimize=True)
        
        # Export TFLite (may fail if TensorFlow not installed)
        tflite_path = os.path.join(output_dir, f"{base_name}.tflite")
        results['tflite'] = self.export_tflite(tflite_path, onnx_path=results['onnx'])
        
        return results
    
    def verify(self, onnx_path):
        """
        Verify ONNX export against original PyTorch model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            bool: True if verification passed
        """
        return verify_onnx(onnx_path, self.dummy_input, self.model)


def get_model_info(model):
    """
    Get comprehensive information about a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model information including size, parameters, and layer count
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    # Count layers
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if layer_type not in layer_counts:
            layer_counts[layer_type] = 0
        layer_counts[layer_type] += 1
    
    return {
        'num_parameters': num_params,
        'num_trainable_parameters': num_trainable,
        'size_bytes': param_bytes + buffer_bytes,
        'size_mb': (param_bytes + buffer_bytes) / (1024 * 1024),
        'layer_counts': layer_counts
    }


def get_file_size(file_path):
    """
    Get the file size in bytes and MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        dict: Size in bytes and MB
    """
    if not os.path.exists(file_path):
        return None
    
    size_bytes = os.path.getsize(file_path)
    return {
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024 * 1024)
    }
