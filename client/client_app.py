import argparse
import time
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import grpc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import service_pb2, service_pb2_grpc
from utils import serialization
from utils.models import SimpleCNN
from client.training import train_one_epoch

def create_dummy_data(num_samples=100):
    # SimpleCNN expects (Batch, 3, 32, 32)
    X = torch.randn(num_samples, 3, 32, 32)
    # 10 classes
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(X, y)

def run_client(client_id, server_address, epochs=5):
    print(f"Starting Client {client_id} connecting to {server_address}")
    
    # 1. Setup Model & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    dataset = create_dummy_data(num_samples=32) # Small dataset for fast testing
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Connect to Server
    channel = grpc.insecure_channel(server_address)
    stub = service_pb2_grpc.FederatedLearningServiceStub(channel)
    
    # 3. Federated Loop
    round_num = 0
    loss = 0.0
    
    try:
        while True:
            round_num += 1
            print(f"\n--- Round {round_num} Start ---")
            
            # A. Serialize Local Parameters
            # Note: For the first round, if we want to pull initial weights, 
            # we could send empty bytes. But simplest is just send random init weights.
            # In production, we might check if round_num == 1 and skip sending.
            
            # If round 1, we might want to get global model first without contributing?
            # Or we contribute our random weights. FedAvg usually starts with a common initialization.
            # Let's try to pull first if it's strictly required, but for convergence 
            # it's better if everyone starts from same point.
            # Here: Send weights -> Server Aggregates -> Returns New Global Weights.
            
            local_params = [p.data.cpu() for p in model.parameters()]
            params_bytes = serialization.parameters_to_bytes(local_params)
            
            fit_request = service_pb2.FitRequest(
                parameters=params_bytes,
                config={
                    "client_id": str(client_id),
                    "num_examples": str(len(dataset)),
                    "loss": str(loss)
                }
            )
            
            print(f"Sending implementation to server ({len(params_bytes)} bytes)...")
            try:
                # This call blocks until server responds (Round complete)
                response = stub.Fit(fit_request)
            except grpc.RpcError as e:
                print(f"RPC failed: {e}")
                time.sleep(5)
                continue
                
            print("Received response from server.")
            
            # B. Update Local Model
            global_params = serialization.bytes_to_parameters(response.parameters)
            if global_params:
                print("Updating local model with global parameters.")
                with torch.no_grad():
                    for p, g_p in zip(model.parameters(), global_params):
                        p.copy_(g_p.to(device))
            
            # C. Local Training
            # train_one_epoch handles FedProx if we pass global_params
            print("Starting local training...")
            loss = train_one_epoch(
                model=model,
                global_model_params=global_params, # For FedProx term
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                mu=0.01
            )
            print(f"Round {round_num} Local Loss: {loss:.4f}")
            
            # Optional: artificial delay to simulate work
            # time.sleep(1)
            
    except KeyboardInterrupt:
        print("Client stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=1, help="Client ID")
    parser.add_argument("--server", type=str, default="localhost:50051", help="Server Address")
    args = parser.parse_args()
    
    run_client(args.id, args.server)
