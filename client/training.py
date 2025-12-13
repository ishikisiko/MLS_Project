import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils import config

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


def train_detection_epoch(model, global_model_params, train_loader, optimizer, device, 
                          mu=0.01, loss_fn=None, input_size=config.INPUT_SIZE):
    """
    Train the detection model for one epoch using FedProx loss.
    
    Args:
        model: The local detection model being trained.
        global_model_params: A list of tensors representing the global model weights.
        train_loader: DataLoader for local detection data.
                     Expected format: (images, targets, paths, shapes)
                     targets: [batch_idx, cls, x, y, w, h]
        optimizer: Local optimizer.
        device: 'cpu' or 'cuda'.
        mu: Proximal term constant (mu) for FedProx.
        loss_fn: Detection loss function (default: DetectionLoss).
        input_size: Input image size for target building.
    
    Returns:
        loss_dict: Dictionary with average loss components.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.detection_loss import DetectionLoss
    
    model.train()
    
    if loss_fn is None:
        loss_fn = DetectionLoss(num_classes=config.NUM_CLASSES)
    
    running_losses = {
        'total_loss': 0.0,
        'box_loss': 0.0,
        'obj_loss': 0.0,
        'cls_loss': 0.0
    }
    num_batches = 0
    
    for batch_data in train_loader:
        # Unpack batch data
        # Format from UADetracDataset: (images, labels, paths, shapes)
        if len(batch_data) == 4:
            images, targets, paths, shapes = batch_data
        elif len(batch_data) == 2:
            images, targets = batch_data
        else:
            raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # (B, N, 4+1+C)
        
        # Detection loss
        det_loss, det_loss_dict = loss_fn(outputs, targets, input_size)
        
        # Add Proximal Term for FedProx
        if global_model_params is not None:
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model_params):
                w_t = w_t.to(device)
                proximal_term += (w - w_t).norm(2) ** 2
            
            total_loss = det_loss + (mu / 2.0) * proximal_term
        else:
            total_loss = det_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (important for detection)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Accumulate losses
        running_losses['total_loss'] += total_loss.item()
        running_losses['box_loss'] += det_loss_dict['box_loss']
        running_losses['obj_loss'] += det_loss_dict['obj_loss']
        running_losses['cls_loss'] += det_loss_dict['cls_loss']
        num_batches += 1
    
    # Average losses
    if num_batches > 0:
        for key in running_losses:
            running_losses[key] /= num_batches
    
    return running_losses


def evaluate_detection(model, val_loader, device, num_classes=4, conf_threshold=0.25):
    """
    Evaluate detection model on validation data.
    
    Args:
        model: Detection model.
        val_loader: Validation DataLoader.
        device: 'cpu' or 'cuda'.
        num_classes: Number of classes.
        conf_threshold: Confidence threshold for predictions.
    
    Returns:
        metrics: Dict with mAP values.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.detection_metrics import DetectionEvaluator
    
    model.eval()
    evaluator = DetectionEvaluator(num_classes=num_classes, conf_threshold=conf_threshold)
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if i % 10 == 0:
                print(f"Eval batch {i}...", end='\r')
            if len(batch_data) == 4:
                images, targets, paths, shapes = batch_data
            elif len(batch_data) == 2:
                images, targets = batch_data
            else:
                continue
            
            images = images.to(device)
            outputs = model(images)
            
            evaluator.update(outputs, targets)
    
    return evaluator.compute()

