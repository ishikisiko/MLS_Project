import grpc
from concurrent import futures
import time
import sys
import os
import threading
import torch
import numpy as np
import copy

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import service_pb2, service_pb2_grpc
from utils import serialization
from utils.models import SimpleCNN
from utils import hardware

# 50MB message size limit for YOLO weights
MAX_MESSAGE_LENGTH = 50 * 1024 * 1024

# Configuration
MIN_FIT_CLIENTS = 2  # Minimum number of clients to wait for before aggregating
ROUND_TIMEOUT = 30.0 # Seconds to wait for clients

class FederatedLearningServicer(service_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self):
        self.round = 1
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Global Model Initialization
        self.global_model = SimpleCNN()
        self.global_model_params = [p.data.clone() for p in self.global_model.parameters()]
        
        # Training State
        self.waiting_updates = [] # List of (parameters_bytes, num_examples, metrics)
        self.ready_for_next_round = False
        
        # Heterogeneous Device Manager
        self.hetero_manager = hardware.HeterogeneousManager()
        
        print("Server initialized.")
        print(f"Waiting for {MIN_FIT_CLIENTS} clients to start Round {self.round}")

    def GetModelForDevice(self, request, context):
        """
        Return appropriate model architecture/config based on device info.
        """
        info = request.device_info
        # Create a temp profile to query
        temp_profile = hardware.DeviceProfile(
            device_id="temp",
            device_type=info.device_type,
            memory_total_mb=info.memory_total_mb, 
            memory_available_mb=info.memory_available_mb,
            cpu_count=info.cpu_count,
            compute_score=info.compute_score
        )
        
        best_model = self.hetero_manager.model_registry.get_best_model_for_device(temp_profile)
        # Fallback
        if not best_model:
            best_model = "SimpleCNN"
            
        print(f"Client requested model recommendation. Score: {info.compute_score}. Recommending: {best_model}")
        
        return service_pb2.GetModelResponse(
            parameters=b"", # Could return initial weights here
            config={"model_name": best_model}
        )

    def Fit(self, request, context):
        """
        Handle a Fit request from a client.
        1. Receive client update (if any).
        2. Wait for other clients / Aggregation.
        3. Return new global model.
        """
        client_id = context.peer()
        
        # --- Hardware-aware: Register Client ---
        if request.HasField('device_info'):
            info = request.device_info
            profile_dict = {
                'device_type': info.device_type,
                'memory_total_mb': info.memory_total_mb,
                'memory_available_mb': info.memory_available_mb, 
                'cpu_count': info.cpu_count,
                'compute_score': info.compute_score
            }
            self.hetero_manager.register_client(client_id, profile_dict)
            print(f"Registered device profile for {client_id}: Score={info.compute_score}")
        # ---------------------------------------
        
        # 1. Collect Update
        with self.lock:
            # Check if this is a fresh start request (empty parameters)
            # For simplicity, we assume clients always send what they have. 
            # If it's the very first join, they might send random or empty weights, 
            # but usually in FL, clients pull initial weights first. 
            # Here keeping it simple: Clients call Fit locally then send.
            # If request.parameters is empty, it's a "pull" request.
            
            has_params = len(request.parameters) > 0
            
            if has_params:
                try:
                    num_examples = int(request.config.get('num_examples', 1))
                except ValueError:
                    num_examples = 1
                    
                print(f"Received update from {client_id}: {num_examples} examples")
                # Parse metrics from config
                metrics = {}
                if 'loss' in request.config:
                    try:
                        metrics['loss'] = float(request.config['loss'])
                    except ValueError:
                        pass
                        
                self.waiting_updates.append({
                    'parameters': request.parameters,
                    'num_examples': num_examples,
                    'metrics': metrics
                })
            else:
                print(f"Client {client_id} connected (requesting initial model)")

            # Check if we can aggregate
            if len(self.waiting_updates) >= MIN_FIT_CLIENTS:
                print(f"Threshold reached ({len(self.waiting_updates)}/{MIN_FIT_CLIENTS}). Aggregating...")
                self._aggregate_models()
                self.waiting_updates = [] # Clear updates for this round
                self.round += 1
                self.ready_for_next_round = True
                self.condition.notify_all() # Wake up all waiting threads
            
            else:
                # Wait for aggregation
                print(f"Waiting for more updates... ({len(self.waiting_updates)}/{MIN_FIT_CLIENTS})")
                
                # We need to capture the current round to know if we advanced
                current_round = self.round
                
                # Wait with timeout
                if not self.condition.wait_for(lambda: self.round > current_round, timeout=ROUND_TIMEOUT):
                    print(f"Timeout waiting for round {current_round} compilation. Returning current model.")
                    # Handle timeout: either abort or proceed with partial.
                    # For now, just return existing global model (dropout/fail logic).
        
        # 3. Return Global Model
        # This part runs after notification or timeout
        # Serialize global model
        serialized_params = serialization.parameters_to_bytes(self.global_model_params)
        
        return service_pb2.FitResponse(
            parameters=serialized_params,
            num_examples=0,
            metrics={"round": float(self.round)}
        )

    def _aggregate_models(self):
        """
        FedAvg: w_global = sum(n_k * w_k) / sum(n_k)
        """
        total_examples = sum(u['num_examples'] for u in self.waiting_updates)
        if total_examples == 0:
            print("No examples for aggregation!")
            return

        print(f"Aggregating {len(self.waiting_updates)} updates with total {total_examples} examples.")
        
        # Initialize zero weights with same shape as global model
        new_weights = [torch.zeros_like(p) for p in self.global_model_params]
        
        # Weighted Sum
        for update in self.waiting_updates:
            client_params = serialization.bytes_to_parameters(update['parameters'])
            n_k = update['num_examples']
            
            for i, param in enumerate(client_params):
                new_weights[i] += param * n_k
        
        # Average
        for i in range(len(new_weights)):
            new_weights[i] = new_weights[i] / total_examples
            
        self.global_model_params = new_weights
        
        # Load back into model object just to keep it synced (optional)
        # self._load_params_to_model(self.global_model, new_weights)
        
        # Monitoring
        avg_loss = 0.0
        updates_with_loss = [u for u in self.waiting_updates if 'loss' in u['metrics']]
        if updates_with_loss:
            avg_loss = sum(u['metrics']['loss'] for u in updates_with_loss) / len(updates_with_loss)
            print(f"Round {self.round} Summary: Aggregated Loss = {avg_loss:.4f}")
            
    def Evaluate(self, request, context):
        return service_pb2.EvaluateResponse(loss=0.0, num_examples=0)

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    service_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        FederatedLearningServicer(), server
    )
    port = '[::]:50051'
    server.add_insecure_port(port)
    print(f"Starting Federated Learning Server on {port}...")
    print(f"Configuration: MIN_FIT_CLIENTS={MIN_FIT_CLIENTS}, TIMEOUT={ROUND_TIMEOUT}s")
    
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
