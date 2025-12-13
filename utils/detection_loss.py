"""
Detection Loss Functions for Object Detection.

Provides CIoU loss for bounding box regression and combined detection loss
that handles objectness and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import config


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4) in format (x, y, w, h)
        box2: Tensor of shape (M, 4) in format (x, y, w, h)
        eps: Small value to avoid division by zero
    
    Returns:
        IoU matrix of shape (N, M)
    """
    # Convert to xyxy format
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area + eps
    
    return inter_area / union_area


def box_ciou(pred, target, eps=1e-7):
    """
    Calculate Complete IoU (CIoU) loss between predicted and target boxes.
    
    CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha * v
    where:
        - rho is the Euclidean distance between box centers
        - c is the diagonal of the smallest enclosing box
        - v measures aspect ratio consistency
        - alpha is a trade-off parameter
    
    Args:
        pred: Predicted boxes (N, 4) in format (x, y, w, h) normalized [0, 1]
        target: Target boxes (N, 4) in format (x, y, w, h) normalized [0, 1]
        eps: Small value to avoid division by zero
    
    Returns:
        CIoU loss value (scalar)
    """
    # Convert to xyxy format
    pred_x1, pred_x2 = pred[:, 0] - pred[:, 2] / 2, pred[:, 0] + pred[:, 2] / 2
    pred_y1, pred_y2 = pred[:, 1] - pred[:, 3] / 2, pred[:, 1] + pred[:, 3] / 2
    target_x1, target_x2 = target[:, 0] - target[:, 2] / 2, target[:, 0] + target[:, 2] / 2
    target_y1, target_y2 = target[:, 1] - target[:, 3] / 2, target[:, 1] + target[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps
    
    # IoU
    iou = inter_area / union_area
    
    # Enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    # Diagonal squared of enclosing box
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
    
    # Center distance squared
    pred_cx, pred_cy = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
    target_cx, target_cy = (target_x1 + target_x2) / 2, (target_y1 + target_y2) / 2
    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    
    # Aspect ratio penalty
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(target[:, 2] / (target[:, 3] + eps)) - torch.atan(pred[:, 2] / (pred[:, 3] + eps)),
        2
    )
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - rho2 / c2 - alpha * v
    
    return 1 - ciou.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (N, C) logits
            target: Target labels (N,) class indices or (N, C) one-hot
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DetectionLoss(nn.Module):
    """
    Combined detection loss for anchor-free object detection.
    
    Total Loss = box_loss + obj_loss + cls_loss
    
    Where:
        - box_loss: CIoU loss for bounding box regression
        - obj_loss: BCE loss for objectness prediction
        - cls_loss: BCE/Focal loss for classification
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES, 
                 box_weight=config.BOX_WEIGHT, 
                 obj_weight=config.OBJ_WEIGHT, 
                 cls_weight=config.CLS_WEIGHT):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Grid anchors will be built dynamically
        self.stride = [8, 16, 32]  # Stride for each feature level
        # Calculate grid sizes based on input size
        self.grid_sizes = [config.INPUT_SIZE // s for s in self.stride]
    
    def build_targets(self, predictions, targets, input_size=config.INPUT_SIZE):
        """
        Build training targets by assigning ground truth to predictions.
        
        Uses a simple center-based assignment strategy:
        GT is assigned to the grid cell containing its center.
        
        Args:
            predictions: Model output (B, N, 4+1+C) where N = sum of all grid cells
            targets: Ground truth (M, 6) format: [batch_idx, cls, x, y, w, h]
            input_size: Input image size
        
        Returns:
            obj_targets: Objectness targets (B, N)
            box_targets: Box targets for positive samples
            cls_targets: Class targets for positive samples
            positive_mask: Boolean mask for positive samples
        """
        device = predictions.device
        batch_size = predictions.size(0)
        num_predictions = predictions.size(1)
        
        # Initialize targets
        obj_targets = torch.zeros(batch_size, num_predictions, device=device)
        box_targets = torch.zeros(batch_size, num_predictions, 4, device=device)
        cls_targets = torch.zeros(batch_size, num_predictions, self.num_classes, device=device)
        
        if targets.numel() == 0:
            return obj_targets, box_targets, cls_targets, obj_targets.bool()
        
        # Build grid cell coordinates for each prediction
        grid_coords = []
        offset = 0
        for stride, grid_size in zip(self.stride, self.grid_sizes):
            x = torch.arange(grid_size, device=device)
            y = torch.arange(grid_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            grid_coords.append((coords, stride, offset, grid_size))
            offset += grid_size * grid_size
        
        # Process each ground truth
        for target in targets:
            batch_idx = int(target[0])
            cls_idx = int(target[1])
            x_center, y_center, w, h = target[2:6]
            
            # Find which grid cell this target falls into
            for coords, stride, offset, grid_size in grid_coords:
                # Scale normalized coords to grid
                gx = (x_center * input_size / stride).long().clamp(0, grid_size - 1)
                gy = (y_center * input_size / stride).long().clamp(0, grid_size - 1)
                grid_idx = gy * grid_size + gx + offset
                
                # Assign target
                obj_targets[batch_idx, grid_idx] = 1.0
                box_targets[batch_idx, grid_idx] = torch.tensor([x_center, y_center, w, h], device=device)
                cls_targets[batch_idx, grid_idx, cls_idx] = 1.0
        
        positive_mask = obj_targets > 0
        return obj_targets, box_targets, cls_targets, positive_mask
    
    def _make_grid(self, input_size, device):
        """Generate grid coordinates for all feature levels."""
        grid_coords = []
        strides = []
        
        for stride in self.stride:
            grid_size = input_size // stride
            x = torch.arange(grid_size, device=device)
            y = torch.arange(grid_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            
            grid_coords.append(coords)
            strides.append(torch.full((coords.shape[0],), stride, device=device))
            
        return torch.cat(grid_coords, dim=0), torch.cat(strides, dim=0)

    def forward(self, predictions, targets, input_size=config.INPUT_SIZE):
        """
        Calculate detection loss.
        
        Args:
            predictions: Model output (B, N, 4+1+C)
            targets: Ground truth (M, 6) format: [batch_idx, cls, x, y, w, h]
            input_size: Input image size
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        device = predictions.device
        batch_size = predictions.size(0)
        
        # Split predictions
        pred_xy = predictions[..., :2]  # (B, N, 2)
        pred_wh = predictions[..., 2:4] # (B, N, 2)
        pred_obj = predictions[..., 4]  # (B, N)
        pred_cls = predictions[..., 5:]  # (B, N, C)
        
        # --- Grid-Relative Decoding ---
        # Get grid coordinates and strides
        grid_coords, strides = self._make_grid(input_size, device)
        
        # Decode XY: (sigmoid(tx) + cx) * stride / input_size
        pred_xy = (torch.sigmoid(pred_xy) + grid_coords) * strides.unsqueeze(-1) / input_size
        
        # Decode WH: sigmoid(tw) (Absolute Normalized - unchanged)
        pred_wh = torch.sigmoid(pred_wh)
        
        # Reassemble boxes
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
        
        # Build targets
        obj_targets, box_targets, cls_targets, positive_mask = self.build_targets(
            predictions, targets, input_size
        )
        
        # Objectness loss (all predictions)
        obj_loss = self.bce_obj(pred_obj, obj_targets)
        
        # Box and class loss (only positive samples)
        num_pos = positive_mask.sum()
        
        if num_pos > 0:
            # Box loss
            pos_pred_box = pred_box[positive_mask]
            pos_box_targets = box_targets[positive_mask]
            box_loss = box_ciou(pos_pred_box, pos_box_targets)
            
            # Class loss
            pos_pred_cls = pred_cls[positive_mask]
            pos_cls_targets = cls_targets[positive_mask]
            cls_loss = self.bce_cls(pos_pred_cls, pos_cls_targets)
        else:
            box_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
        
        # Weighted sum
        total_loss = (
            self.box_weight * box_loss +
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss
        )
        
        loss_dict = {
            'box_loss': box_loss.item() if isinstance(box_loss, torch.Tensor) else box_loss,
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'total_loss': total_loss.item(),
            'num_pos': num_pos.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test loss computation
    batch_size = 2
    num_predictions = 8400
    num_classes = config.NUM_CLASSES
    
    # Dummy predictions
    predictions = torch.randn(batch_size, num_predictions, 4 + 1 + num_classes)
    
    # Dummy targets: [batch_idx, cls, x, y, w, h]
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.3],  # Batch 0, class 1
        [0, 2, 0.3, 0.7, 0.1, 0.2],  # Batch 0, class 2
        [1, 0, 0.6, 0.4, 0.15, 0.25], # Batch 1, class 0
    ])
    
    loss_fn = DetectionLoss(num_classes=num_classes)
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss Details: {loss_dict}")
