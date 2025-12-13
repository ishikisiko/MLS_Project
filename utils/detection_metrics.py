"""
Detection Metrics for Object Detection Evaluation.

Provides mAP computation at various IoU thresholds following COCO evaluation protocol.
"""

import torch
import numpy as np
from collections import defaultdict
from utils import config


def box_iou_numpy(box1, box2):
    """
    Calculate IoU between two sets of boxes (numpy version).
    
    Args:
        box1: Array of shape (N, 4) in format (x1, y1, x2, y2)
        box2: Array of shape (M, 4) in format (x1, y1, x2, y2)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    # Intersection
    x1 = np.maximum(box1[:, 0][:, None], box2[:, 0])
    y1 = np.maximum(box1[:, 1][:, None], box2[:, 1])
    x2 = np.minimum(box1[:, 2][:, None], box2[:, 2])
    y2 = np.minimum(box1[:, 3][:, None], box2[:, 3])
    
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-7)


def xywh_to_xyxy(boxes):
    """
    Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2).
    
    Args:
        boxes: Array/Tensor of shape (N, 4)
    
    Returns:
        Converted boxes of shape (N, 4)
    """
    if isinstance(boxes, torch.Tensor):
        result = boxes.clone()
    else:
        result = boxes.copy()
    
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    return result


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using 101-point interpolation.
    
    Args:
        recall: Recall values array
        precision: Precision values array
    
    Returns:
        AP value
    """
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        # Find precision at this recall
        prec = mpre[mrec >= r]
        ap += prec[0] if len(prec) > 0 else 0.0
    
    return ap / 101


def compute_ap_per_class(predictions, targets, iou_threshold=config.IOU_THRESHOLD, num_classes=config.NUM_CLASSES):
    """
    Compute AP for each class at given IoU threshold.
    
    Args:
        predictions: List of dicts with keys 'boxes', 'scores', 'labels'
            - boxes: (N, 4) in xyxy format
            - scores: (N,) confidence scores
            - labels: (N,) class labels
        targets: List of dicts with keys 'boxes', 'labels'
            - boxes: (M, 4) in xyxy format
            - labels: (M,) class labels
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        ap_per_class: Dict mapping class_id to AP value
        metrics: Additional metrics dict
    """
    ap_per_class = {}
    
    for cls in range(num_classes):
        # Collect all predictions and targets for this class
        all_preds = []  # (image_idx, box, score)
        all_targets = defaultdict(list)  # image_idx -> list of boxes
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Predictions for this class
            if pred['labels'] is not None and len(pred['labels']) > 0:
                mask = pred['labels'] == cls
                if mask.any():
                    cls_boxes = pred['boxes'][mask]
                    cls_scores = pred['scores'][mask]
                    for box, score in zip(cls_boxes, cls_scores):
                        all_preds.append((img_idx, box, float(score)))
            
            # Targets for this class
            if target['labels'] is not None and len(target['labels']) > 0:
                mask = target['labels'] == cls
                if mask.any():
                    all_targets[img_idx] = target['boxes'][mask]
        
        # Count total GT
        n_gt = sum(len(boxes) for boxes in all_targets.values())
        
        if n_gt == 0:
            ap_per_class[cls] = 0.0
            continue
        
        if len(all_preds) == 0:
            ap_per_class[cls] = 0.0
            continue
        
        # Sort by score (descending)
        all_preds.sort(key=lambda x: x[2], reverse=True)
        
        # Track which GT boxes have been matched
        matched = {img_idx: np.zeros(len(boxes), dtype=bool) 
                   for img_idx, boxes in all_targets.items()}
        
        # Compute TP/FP
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        for pred_idx, (img_idx, pred_box, score) in enumerate(all_preds):
            if img_idx not in all_targets:
                fp[pred_idx] = 1
                continue
            
            gt_boxes = all_targets[img_idx]
            if len(gt_boxes) == 0:
                fp[pred_idx] = 1
                continue
            
            # Calculate IoU with all GT boxes
            ious = box_iou_numpy(pred_box.reshape(1, 4), gt_boxes)[0]
            
            # Find best matching GT
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            
            if best_iou >= iou_threshold and not matched[img_idx][best_iou_idx]:
                tp[pred_idx] = 1
                matched[img_idx][best_iou_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap_per_class[cls] = compute_ap(recall, precision)
    
    return ap_per_class


def compute_map(predictions, targets, num_classes=config.NUM_CLASSES, iou_thresholds=None):
    """
    Compute mAP at various IoU thresholds.
    
    Args:
        predictions: List of prediction dicts per image
        targets: List of target dicts per image
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds (default: [0.5] and [0.5:0.95:0.05])
    
    Returns:
        metrics: Dict with mAP values
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # Compute AP@0.5
    ap_50 = compute_ap_per_class(predictions, targets, 0.5, num_classes)
    map_50 = np.mean(list(ap_50.values()))
    
    # Compute mAP@[0.5:0.95]
    all_aps = []
    for iou_thresh in iou_thresholds:
        ap_dict = compute_ap_per_class(predictions, targets, iou_thresh, num_classes)
        all_aps.append(np.mean(list(ap_dict.values())))
    
    map_50_95 = np.mean(all_aps)
    
    return {
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_50_95,
        'AP_per_class@0.5': ap_50
    }


class DetectionEvaluator:
    """
    Evaluator for object detection models.
    
    Collects predictions and targets, then computes mAP metrics.
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES, iou_threshold=config.IOU_THRESHOLD, conf_threshold=config.CONF_THRESHOLD):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.stride = [8, 16, 32]  # Stride for each feature level
        self.reset()
    
    def reset(self):
        """Reset collected predictions and targets."""
        self.predictions = []
        self.targets = []

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
    
    def decode_predictions(self, outputs, conf_threshold=None, input_size=config.INPUT_SIZE):
        """
        Decode raw model outputs to boxes, scores, labels.
        
        Args:
            outputs: Model output (B, N, 4+1+C)
            conf_threshold: Confidence threshold for filtering
            input_size: Input image size
        
        Returns:
            List of prediction dicts per image
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        batch_preds = []
        batch_size = outputs.size(0)
        device = outputs.device
        
        # Grid-Relative Decoding
        grid_coords, strides = self._make_grid(input_size, device)
        
        for b in range(batch_size):
            pred = outputs[b]  # (N, 4+1+C)
            
            # Split predictions
            pred_xy = pred[:, :2]
            pred_wh = pred[:, 2:4]
            obj_conf = torch.sigmoid(pred[:, 4])
            cls_conf = torch.sigmoid(pred[:, 5:])
            
            # Decode XY: (sigmoid(tx) + cx) * stride / input_size
            # Note: We do this before filtering to ensure correct broadcasting, 
            # but for efficiency we could filter first. 
            # However, grid_coords matches N, so filtering first requires filtering grid too.
            
            # Overall confidence
            scores, labels = cls_conf.max(dim=1)
            scores = scores * obj_conf
            
            # Filter by confidence
            mask = scores > conf_threshold
            
            if mask.any():
                # Filter first to save computation
                pred_xy = pred_xy[mask]
                pred_wh = pred_wh[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                curr_grid = grid_coords[mask]
                curr_stride = strides[mask]
                
                # Decode
                decoded_xy = (torch.sigmoid(pred_xy) + curr_grid) * curr_stride.unsqueeze(-1) / input_size
                decoded_wh = torch.sigmoid(pred_wh)
                
                boxes = torch.cat([decoded_xy, decoded_wh], dim=-1)
                
                # Convert to xyxy
                boxes = xywh_to_xyxy(boxes)
                
                batch_preds.append({
                    'boxes': boxes.cpu().numpy(),
                    'scores': scores.cpu().numpy(),
                    'labels': labels.cpu().numpy()
                })
            else:
                batch_preds.append({
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'labels': np.array([])
                })
        
        return batch_preds
    
    def process_targets(self, targets, batch_size):
        """
        Process targets from dataloader format to evaluation format.
        
        Args:
            targets: Tensor (M, 6) format [batch_idx, cls, x, y, w, h]
            batch_size: Batch size (needed to handle images with no objects)
        
        Returns:
            List of target dicts per image in batch
        """
        batch_targets = []
        
        for b in range(batch_size):
            # Find targets for this batch index
            mask = targets[:, 0] == b
            if mask.any():
                t = targets[mask]
                boxes = xywh_to_xyxy(t[:, 2:6])
                batch_targets.append({
                    'boxes': boxes.cpu().numpy(),
                    'labels': t[:, 1].long().cpu().numpy()
                })
            else:
                batch_targets.append({
                    'boxes': np.array([]).reshape(0, 4),
                    'labels': np.array([])
                })
        
        return batch_targets
    
    def update(self, outputs, targets):
        """
        Add batch of predictions and targets.
        
        Args:
            outputs: Model output (B, N, 4+1+C)
            targets: Ground truth (M, 6)
        """
        batch_size = outputs.size(0)
        preds = self.decode_predictions(outputs)
        tgts = self.process_targets(targets, batch_size)
        
        self.predictions.extend(preds)
        self.targets.extend(tgts)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            metrics: Dict with mAP values
        """
        if len(self.predictions) == 0:
            return {
                'mAP@0.5': 0.0,
                'mAP@0.5:0.95': 0.0,
                'AP_per_class@0.5': {i: 0.0 for i in range(self.num_classes)}
            }
        
        return compute_map(self.predictions, self.targets, self.num_classes)


if __name__ == '__main__':
    # Test metrics computation
    evaluator = DetectionEvaluator(num_classes=config.NUM_CLASSES)
    
    # Dummy predictions
    outputs = torch.randn(2, 100, 4 + 1 + config.NUM_CLASSES)  # 2 images, 100 predictions, 9 outputs
    
    # Dummy targets
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.3],
        [0, 2, 0.3, 0.7, 0.1, 0.2],
        [1, 0, 0.6, 0.4, 0.15, 0.25],
    ])
    
    evaluator.update(outputs, targets)
    metrics = evaluator.compute()
    
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"AP per class: {metrics['AP_per_class@0.5']}")
