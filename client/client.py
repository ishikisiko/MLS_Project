import grpc
import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import service_pb2, service_pb2_grpc
from utils import serialization, privacy, hardware

MAX_MESSAGE_LENGTH = 50 * 1024 * 1024 

def run_client():
    server_address = 'localhost:50051'
    
    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
    
    print(f"Connecting to server at {server_address}...")
    with grpc.insecure_channel(server_address, options=channel_options) as channel:
        stub = service_pb2_grpc.FederatedLearningServiceStub(channel)
        
        # --- Hardware-aware Optimization: Profiling ---
        profiler = hardware.DeviceProfiler()
        profile = profiler.detect_device()
        print(f"Device Profile: {profile}")
        
        scheduler = hardware.ResourceScheduler()
        
        # Prepare DeviceInfo proto
        device_info_proto = service_pb2.DeviceInfo(
            device_type=profile.device_type,
            memory_total_mb=profile.memory_total_mb,
            memory_available_mb=profile.memory_available_mb,
            compute_score=profile.compute_score,
            cpu_count=profile.cpu_count
        )
        # ----------------------------------------------

        # Initialize DP Engine
        # Epsilon/Delta management would typically be here or tracked by an accountant
        dp_engine = privacy.DPEngine(max_norm=1.0, noise_multiplier=0.1)

        # Initialize Model and Training components
        from utils import models
        from client import training
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Hardware-aware Optimization: Model Selection ---
        # Try to get appropriate model from server based on device capabilities
        model_name = "SimpleCNN" # Default
        try:
            get_model_req = service_pb2.GetModelRequest(device_info=device_info_proto)
            # Commented out until server implements this handler fully to avoid crash
            # get_model_resp = stub.GetModelForDevice(get_model_req)
            # if get_model_resp.config.get('model_name'):
            #     model_name = get_model_resp.config['model_name']
            pass
        except grpc.RpcError:
            print("Server does not support GetModelForDevice yet, using default.")
        
        print(f"Using model: {model_name}")
        model = models.SimpleCNN().to(device)
        # ----------------------------------------------------
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Global weights storage (initially None or matching local)
        global_weights = [p.clone().detach() for p in model.parameters()]

        # Simulate FedAvg Rounds
        for round_num in range(1, 4):
            print(f"\n--- Round {round_num} ---")
            
            # --- Hardware-aware Optimization: Dynamic Batch Size ---
            # Estimate model size (approx 0.5MB for SimpleCNN)
            model_size_mb = 0.5 
            optimal_batch_size = scheduler.compute_optimal_batch_size(profile, model_size_mb)
            print(f"Optimal Batch Size for this device: {optimal_batch_size}")
            # -------------------------------------------------------

            # 1. Simulate Data (since we don't have dataset yet)
            # Create random tensors for 32x32 images
            dummy_inputs = torch.randn(optimal_batch_size * 2, 3, 32, 32)
            dummy_labels = torch.randint(0, 10, (optimal_batch_size * 2,))
            train_dataset = torch.utils.data.TensorDataset(dummy_inputs, dummy_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)

            # --- Hardware-aware Optimization: Adaptive Local Epochs ---
            # Estimate time per step (e.g., 100ms) or measure it
            local_epochs = scheduler.compute_adaptive_epochs(profile, latency_budget_ms=2000.0) 
            print(f"Adaptive Local Epochs: {local_epochs}")
            # ----------------------------------------------------------

            # 2. Train locally (FedProx)
            print(f"Training locally with FedProx for {local_epochs} epochs...")
            epoch_loss = 0.0
            for _ in range(local_epochs):
                loss = training.train_one_epoch(
                    model, 
                    global_weights, 
                    train_loader, 
                    optimizer, 
                    device, 
                    mu=0.1
                )
                epoch_loss += loss
            
            # Average loss over epochs
            epoch_loss /= local_epochs
            print(f"Round {round_num} finished. Average Loss: {epoch_loss:.4f}")
            
            # 3. Get gradients/update for DP
            # In FedAvg, update = w_new - w_old (or just w_new if server does aggregation differently)
            # Here we send the model parameters directly, but for DP we typically clip gradients.
            # However, in FL-DP, we often clip the *update* (delta).
            
            new_weights = [p.data.clone() for p in model.parameters()]
            
            # Calculate delta: w_new - w_old (global)
            # CAUTION: If global_weights is None or stale, this might be wrong. 
            # Ideally we receive global weights from server at start of round.
            # For simulation, we assume we update `model` in place starting from `global_weights`.
            
            model_delta = [new - old.to(device) for new, old in zip(new_weights, global_weights)]
            
            # 4. Apply Differential Privacy (Clipping + Noise on Delta)
            print("Applying Differential Privacy (Clipping + Noise)...")
            dp_update = dp_engine.step(model_delta)
            
            # Reconstruct "noisy" new weights to send (w_old + noisy_delta) 
            # OR send noisy delta directly. Protocol expects `parameters`.
            # If server aggregates weights: send w_old + dp_update
            # If server aggregates deltas: send dp_update
            # Let's assume standard FedAvg expects weights.
            
            noisy_weights = [old.to(device) + update for old, update in zip(global_weights, dp_update)]
            
            # 5. Serialize
            local_update_bytes = serialization.parameters_to_bytes(noisy_weights)
            
            # 6. Send to Server (FitRequest)
            print("Sending update to server...")
            try:
                request = service_pb2.FitRequest(
                    parameters=local_update_bytes,
                    config={"round": str(round_num)},
                    device_info=device_info_proto  # Sending device info
                )
                response = stub.Fit(request)
                
                print(f"Received response from server. Global model size: {len(response.parameters)} bytes")
                
                # Update global weights for next round
                # In real execution, we deserialize response.parameters
                if response.parameters:
                    # If server returns updated global model (it should in FedAvg)
                    try:
                        loaded_weights = serialization.bytes_to_parameters(response.parameters)
                        # loaded_weights might be state_dict or list of params depending on server implementation.
                        # Our server skeleton is very basic right now, likely echoing.
                        # For this step, we'll try to load it. 
                        # If server is just an echo or empty, we keep our own reference for simulation continuity.
                        if isinstance(loaded_weights, list):
                             global_weights = loaded_weights
                    except Exception as e:
                        print(f"Failed to load global weights from server: {e}")

            except grpc.RpcError as e:
                print(f"RPC Error: {e.code()} - {e.details()}")
                break

if __name__ == '__main__':
    run_client()
