import grpc
import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import service_pb2, service_pb2_grpc
from utils import serialization, privacy, hardware, config, adaptive

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

MAX_MESSAGE_LENGTH = config.MAX_MESSAGE_LENGTH 

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


def run_detection_client(data_root=None, num_rounds=3):
    """
    Run federated learning client for object detection task.
    Uses real UA-DETRAC dataset and YOLOv11n detection model.
    """
    server_address = 'localhost:50051'
    
    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
    
    print(f"Connecting to server at {server_address}...")
    print("Running Detection Client with UA-DETRAC dataset and YOLOv11n model")
    
    with grpc.insecure_channel(server_address, options=channel_options) as channel:
        stub = service_pb2_grpc.FederatedLearningServiceStub(channel)
        
        # Hardware profiling
        profiler = hardware.DeviceProfiler()
        profile = profiler.detect_device()
        print(f"Device Profile: {profile}")
        
        scheduler = hardware.ResourceScheduler()
        
        device_info_proto = service_pb2.DeviceInfo(
            device_type=profile.device_type,
            memory_total_mb=profile.memory_total_mb,
            memory_available_mb=profile.memory_available_mb,
            compute_score=profile.compute_score,
            cpu_count=profile.cpu_count
        )
        
        # Initialize DP Engine
        dp_engine = privacy.DPEngine(max_norm=1.0, noise_multiplier=0.1)

        # --- Adaptive System Initialization ---
        network_monitor = adaptive.NetworkMonitor()
        device_monitor = adaptive.DeviceAvailability()
        adaptive_strategy = adaptive.AdaptiveStrategy(network_monitor, device_monitor)
        print("Adaptive System initialized.")
        # --------------------------------------
        
        # Import detection model and data loader
        from utils.detection_models import YOLOv11n
        from utils.data_loader import get_ua_detrac_loaders
        from client import training
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize Detection Model
        print("Initializing YOLOv11n detection model...")
        model = YOLOv11n(num_classes=config.NUM_CLASSES).to(device)
        print(f"Model parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        # Global weights storage
        global_weights = [p.clone().detach() for p in model.parameters()]
        
        # Load Data
        print("Loading UA-DETRAC dataset...")
        model_size_mb = model.get_num_params() * 4 / (1024 * 1024)  # Float32
        optimal_batch_size = min(scheduler.compute_optimal_batch_size(profile, model_size_mb), 8)
        optimal_batch_size = max(optimal_batch_size, 2)  # At least batch size 2
        
        if data_root is None:
            data_root = config.DEFAULT_DATA_ROOT
            
        train_loader, val_loader = get_ua_detrac_loaders(
            data_root=data_root,
            batch_size=optimal_batch_size,
            num_workers=0
        )
        
        print(f"Optimal Batch Size: {optimal_batch_size}")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        
        # Federated Learning Rounds
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*50}")
            print(f"Round {round_num}/{num_rounds}")
            print(f"{'='*50}")
            
            # Adaptive local epochs
            local_epochs = scheduler.compute_adaptive_epochs(profile, latency_budget_ms=5000.0)
            local_epochs = max(local_epochs, 1)
            print(f"Local Epochs: {local_epochs}")
            
            # Start timing for monitoring
            training_start_time = time.time()
            
            # --- Adaptive Check & Parameters ---
            # 1. Check participation
            if config.ADAPTIVE_ENABLED and not adaptive_strategy.should_participate():
                print(f"Skipping Round {round_num} due to poor network or device availability.")
                print(f"  Network Score: {network_monitor.get_network_quality_score():.1f}")
                print(f"  Availability: {device_monitor.get_availability_score():.1f}")
                # Notify server of skip? Or just silence. 
                # For FL, silence usually means dropout. check timeout.
                # To be polite, we could send an empty update or just wait.
                # Here we just continue (dropout).
                continue

            # 2. Get Adaptive Parameters
            if config.ADAPTIVE_ENABLED:
                adaptive_params = adaptive_strategy.compute_adaptive_params(
                    base_batch_size=optimal_batch_size,
                    base_epochs=scheduler.compute_adaptive_epochs(profile, latency_budget_ms=5000.0)
                )
                
                # Apply adaptive parameters
                current_batch_size = adaptive_params.batch_size
                local_epochs = adaptive_params.local_epochs
                compression_level = adaptive_params.compression_level
                
                print(f"Adaptive Parameters: Batch={current_batch_size}, Epochs={local_epochs}, Compression={compression_level}")
            else:
                current_batch_size = optimal_batch_size
                # local_epochs already set above
                compression_level = 'none'

            # Update DataLoader if batch size changed
            if current_batch_size != train_loader.batch_size:
                print(f"Adjusting batch size to {current_batch_size}...")
                train_loader, _ = get_ua_detrac_loaders(
                    data_root=data_root,
                    batch_size=current_batch_size,
                    num_workers=0
                )
            # -----------------------------------
            
            # Train locally (FedProx with detection loss)
            print("Training locally with FedProx + Detection Loss...")
            total_losses = {'total_loss': 0, 'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0}
            
            for epoch in range(local_epochs):
                epoch_losses = training.train_detection_epoch(
                    model,
                    global_weights,
                    train_loader,
                    optimizer,
                    device,
                    mu=config.FEDPROX_MU,
                    input_size=config.INPUT_SIZE
                )
                for k in total_losses:
                    total_losses[k] += epoch_losses[k]
                print(f"  Epoch {epoch+1}/{local_epochs}: Loss={epoch_losses['total_loss']:.4f} "
                      f"(box={epoch_losses['box_loss']:.4f}, obj={epoch_losses['obj_loss']:.4f}, cls={epoch_losses['cls_loss']:.4f})")
            
            # Calculate training duration
            training_duration = time.time() - training_start_time
            
            # Collect resource usage for monitoring
            memory_usage_mb = 0.0
            cpu_percent = 0.0
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=0.1)
            
            print(f"Training metrics: duration={training_duration:.2f}s, memory={memory_usage_mb:.1f}MB, cpu={cpu_percent:.1f}%")
            
            # Average losses
            for k in total_losses:
                total_losses[k] /= local_epochs
            
            print(f"Round {round_num} training complete. Avg Loss: {total_losses['total_loss']:.4f}")
            
            # Get model updates for DP
            new_weights = [p.data.clone() for p in model.parameters()]
            model_delta = [new - old.to(device) for new, old in zip(new_weights, global_weights)]
            
            # Apply Differential Privacy
            print("Applying Differential Privacy...")
            dp_update = dp_engine.step(model_delta)
            noisy_weights = [old.to(device) + update for old, update in zip(global_weights, dp_update)]
            
            # Serialize and send to server
            local_update_bytes = serialization.parameters_to_bytes(noisy_weights)
            
            print(f"Sending update to server ({len(local_update_bytes) / 1024 / 1024:.2f} MB)...")
            try:
                request = service_pb2.FitRequest(
                    parameters=local_update_bytes,
                    config={
                        "round": str(round_num),
                        "loss": str(total_losses['total_loss']),
                        "num_examples": str(len(train_loader) * optimal_batch_size),
                        "training_duration": str(training_duration),
                        "memory_usage_mb": str(memory_usage_mb),
                        "cpu_percent": str(cpu_percent),
                        # Adaptive Metrics
                        "network_quality": str(network_monitor.get_network_quality_score()),
                        "availability_score": str(device_monitor.get_availability_score()),
                        "compression_level": compression_level
                    },
                    device_info=device_info_proto
                )
                response = stub.Fit(request)
                
                print(f"Received global model from server ({len(response.parameters)} bytes)")
                
                # Update global weights
                if response.parameters:
                    try:
                        loaded_weights = serialization.bytes_to_parameters(response.parameters)
                        if isinstance(loaded_weights, list):
                            global_weights = loaded_weights
                    except Exception as e:
                        print(f"Failed to load global weights: {e}")
                        
            except grpc.RpcError as e:
                print(f"RPC Error: {e.code()} - {e.details()}")
                break
        
        print("\n" + "="*50)
        print("Federated Learning Complete!")
        print("="*50)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--mode', type=str, default='detection', choices=['classification', 'detection'],
                        help='Training mode: classification (old) or detection (new)')
    parser.add_argument('--data-root', type=str, default=None, 
                        help='Path to UA-DETRAC dataset root')
    parser.add_argument('--rounds', type=int, default=3, help='Number of FL rounds')
    args = parser.parse_args()
    
    if args.mode == 'detection':
        run_detection_client(data_root=args.data_root, num_rounds=args.rounds)
    else:
        run_client()

