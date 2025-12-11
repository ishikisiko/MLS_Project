import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_one_epoch(model, global_model_params, train_loader, optimizer, device, mu=0.01):
    """
    Train the model for one epoch using FedProx loss.
    
    Args:
        model: The local model being trained.
        global_model_params: A list of tensors representing the global model weights.
                             Must match the order of model.parameters().
        train_loader: DataLoader for local data.
        optimizer: Local optimizer.
        device: 'cpu' or 'cuda'.
        mu: Proximal term constant (mu).
    
    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # 1. Calculate standard loss (CrossEntropy)
        loss = criterion(outputs, labels)
        
        # 2. Add Proximal Term for FedProx
        # L_prox = (mu / 2) * || w - w_t ||^2
        if global_model_params is not None:
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model_params):
                # Ensure w_t is on the same device
                w_t = w_t.to(device)
                proximal_term += (w - w_t).norm(2) ** 2
            
            loss += (mu / 2.0) * proximal_term
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)
