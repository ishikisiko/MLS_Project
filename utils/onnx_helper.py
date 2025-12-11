import onnxruntime as ort
import numpy as np
import os

class ONNXPredictor:
    """
    A helper class to load and run ONNX models using ONNX Runtime.
    """
    def __init__(self, model_path, use_gpu=False):
        """
        Initialize the ONNXPredictor.

        Args:
            model_path (str): Path to the .onnx model file.
            use_gpu (bool): Whether to use GPU (CUDAExecutionProvider) if available.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")

        self.model_path = model_path
        
        providers = ['CPUExecutionProvider']
        if use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            else:
                print("Warning: CUDAExecutionProvider requested but not available. Using CPU.")

        print(f"Loading ONNX model from {model_path} with providers: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        """
        Run inference on the input data.

        Args:
            input_data (numpy.ndarray): Input data matching the model's input shape.

        Returns:
            numpy.ndarray: Model output.
        """
        # Ensure input is numpy array
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        # Basic shape check if possible (dynamic axes might make shape contain strings or -1)
        # Verify dtype
        input_type = self.session.get_inputs()[0].type
        if 'float' in input_type and input_data.dtype != np.float32:
             input_data = input_data.astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]
