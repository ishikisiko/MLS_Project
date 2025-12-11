import grpc
from concurrent import futures
import time
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import service_pb2, service_pb2_grpc
from utils import serialization

# 50MB message size limit for YOLO weights
MAX_MESSAGE_LENGTH = 50 * 1024 * 1024 

class FederatedLearningServicer(service_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self):
        self.round = 0
        # In a real scenario, we would load the initial global model here
        # self.global_model = load_model()
        print("Server initialized.")

    def Fit(self, request, context):
        print(f"Received Fit request from client.")
        # Deserialize client parameters (if any) or just acknowledge
        # For the first round, the client might be asking for the model, 
        # or sending its update. 
        # Here we assume standard FL: Model is pulled or sent?
        # Typically Client calls Fit(ClientParams) -> Server returns (GlobalParams)
        # OR Server calls Client (if bidirectional stream).
        # But our proto defines: rpc Fit (FitRequest) returns (FitResponse);
        # FitRequest usually comes from Client? 
        # If Client is the initiator (gRPC Client), it sends FitRequest.
        # So FitRequest contains Client's update? No, usually Client requests "Instructions"
        # or Client sends "Update".
        # Let's assume:
        # Client sends FitRequest with its LOCAL weights (if it has them) OR empty if asking for init.
        # Server returns FitResponse with GLOBAL weights.
        
        # However, typically in Flower/FedML:
        # Server -> Client: FitIns (Parameters)
        # Client -> Server: FitRes (Parameters)
        
        # If we use gRPC where Client connects to Server:
        # Client calls Fit(LocalUpdate) -> Returns GlobalModel
        
        # In the context of this project, let's implement a simple logic:
        # Request contains "parameters" -> Client's update
        # Response contains "parameters" -> New Global Model
        
        client_params_bytes = request.parameters
        if len(client_params_bytes) > 0:
            print(f"Received update of size {len(client_params_bytes)} bytes")
            # In real logic: Aggregate(client_params)
            
        # Return global model (dummy for now)
        # In real logic: Serialize self.global_model
        dummy_global_model_bytes = b'DUMMY_GLOBAL_MODEL_WEIGHTS'
        
        return service_pb2.FitResponse(
            parameters=dummy_global_model_bytes,
            num_examples=0, # Just a config/signal
            metrics={"accept": 1.0}
        )

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
    print(f"Starting server on {port}...")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
