import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.onnx_helper import ONNXPredictor

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def create_dummy_onnx_model(path):
    """Creates a simple pytorch model and exports it to ONNX."""
    model = SimpleModel()
    model.eval()
    
    # Dummy input for export
    dummy_input = torch.randn(1, 10)
    
    print(f"Exporting dummy ONNX model to {path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        path, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

def run_baseline_inference():
    model_path = "baseline_model.onnx"
    
    # 1. Create a dummy model if it doesn't exist (for testing purposes)
    if not os.path.exists(model_path):
        create_dummy_onnx_model(model_path)
    
    # 2. Initialize Predictor
    try:
        predictor = ONNXPredictor(model_path)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    # 3. Prepare Dummy Data
    # Batch size 5, input features 10
    input_data = np.random.randn(5, 10).astype(np.float32)
    
    # 4. Run Inference
    print("\nRunning inference...")
    try:
        output = predictor.predict(input_data)
        print("Inference successful!")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
        print("Output values sample:\n", output)
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    run_baseline_inference()
