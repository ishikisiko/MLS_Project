"""
Lightweight Object Detection Models for Federated Learning.

Provides YOLOv11n-style architecture optimized for vehicle detection
on edge devices with limited compute resources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.quantized import FloatFunctional
from utils import config

# Class names for UA-DETRAC dataset
NAMES = config.CLASS_NAMES


class ConvBlock(nn.Module):
    """Standard Conv + BatchNorm + SiLU activation block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) # Changed from SiLU to ReLU for quantization support
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut."""
    
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden, 1)
        self.cv2 = ConvBlock(hidden, out_channels, 3)
        self.add = shortcut and in_channels == out_channels
        self.skip_add = FloatFunctional()
    
    def forward(self, x):
        return self.skip_add.add(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""
    
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, 2 * self.c, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, expansion=1.0) for _ in range(n)
        )
        self.ff = FloatFunctional()
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.ff.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)."""
    
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden, 1)
        self.cv2 = ConvBlock(hidden * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.ff = FloatFunctional()
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.ff.cat([x, y1, y2, self.m(y2)], 1))


class DetectionHead(nn.Module):
    """
    Detection head for anchor-free detection.
    
    Outputs predictions in format: (batch, num_predictions, 4 + 1 + num_classes)
    Where: 4 = box coords (x, y, w, h), 1 = objectness, num_classes = class probabilities
    """
    
    def __init__(self, in_channels_list, num_classes=config.NUM_CLASSES, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = 4 + 1 + num_classes  # box + obj + cls
        
        # Separate heads for each feature level
        self.heads = nn.ModuleList()
        for in_ch in in_channels_list:
            head = nn.Sequential(
                ConvBlock(in_ch, in_ch, 3),
                ConvBlock(in_ch, in_ch, 3),
                nn.Conv2d(in_ch, self.num_outputs, 1)
            )
            self.heads.append(head)
        
        # Initialize biases
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize detection biases for faster convergence."""
        with torch.no_grad():
            for head in self.heads:
                # Get the final conv layer
                final_conv = head[-1]
                b = final_conv.bias.view(-1)
                # Objectness bias (assume 1/1000 objects per cell)
                b[4] = math.log(1 / (config.INPUT_SIZE / 8) ** 2)  # obj
                # Class bias
                b[5:] = math.log(0.25 / (self.num_classes - 0.25))
                final_conv.bias.copy_(b.view(-1))
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [P3, P4, P5]
        
        Returns:
            Tensor of shape (batch, total_anchors, 4 + 1 + num_classes)
        """
        outputs = []
        for feat, head in zip(features, self.heads):
            out = head(feat)  # (B, num_outputs, H, W)
            B, _, H, W = out.shape
            out = out.view(B, self.num_outputs, -1).permute(0, 2, 1)  # (B, H*W, num_outputs)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)  # (B, total_anchors, num_outputs)


class YOLOv11n(nn.Module):
    """
    YOLOv11 Nano - Lightweight detection model for edge deployment.
    
    Designed for 640x640 input with 4 vehicle classes (UA-DETRAC).
    Output: (batch, 8400, 9) where 9 = 4 box + 1 obj + 4 cls
    
    Architecture:
    - Backbone: Efficient feature extraction with C2f blocks
    - Neck: FPN-style multi-scale feature fusion
    - Head: Anchor-free detection head
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES, input_size=config.INPUT_SIZE, width_mult=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Channel scaler
        def c(x): return int(x * width_mult)
        
        # Backbone channels: [32, 64, 128, 256] * width_mult
        
        # ===== Backbone =====
        # Stem
        self.stem = ConvBlock(3, c(16), 3, 2)  # 640 -> 320
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(c(16), c(32), 3, 2),  # 320 -> 160
            C2f(c(32), c(32), n=1)
        )
        
        # Stage 2 (P3 - 80x80)
        self.stage2 = nn.Sequential(
            ConvBlock(c(32), c(64), 3, 2),  # 160 -> 80
            C2f(c(64), c(64), n=2)
        )
        
        # Stage 3 (P4 - 40x40)
        self.stage3 = nn.Sequential(
            ConvBlock(c(64), c(128), 3, 2),  # 80 -> 40
            C2f(c(128), c(128), n=2)
        )
        
        # Stage 4 (P5 - 20x20)
        self.stage4 = nn.Sequential(
            ConvBlock(c(128), c(256), 3, 2),  # 40 -> 20
            C2f(c(256), c(256), n=1),
            SPPF(c(256), c(256))
        )
        
        # ===== Neck (FPN) =====
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P5 -> P4
        self.lateral_p5 = ConvBlock(c(256), c(128), 1)
        self.fpn_p4 = C2f(c(256), c(128), n=1, shortcut=False)  # 128(up) + 128(backbone) = 256 in
        
        # P4 -> P3  
        self.lateral_p4 = ConvBlock(c(128), c(64), 1)
        self.fpn_p3 = C2f(c(128), c(64), n=1, shortcut=False)  # 64(up) + 64(backbone) = 128 in
        
        # Bottom-up path (PAN)
        # P3 -> P4
        self.downsample_p3 = ConvBlock(c(64), c(64), 3, 2)
        self.pan_p4 = C2f(c(192), c(128), n=1, shortcut=False)  # 64(down) + 128(lateral) = 192 in
        
        # P4 -> P5
        self.downsample_p4 = ConvBlock(c(128), c(128), 3, 2)
        self.pan_p5 = C2f(c(384), c(256), n=1, shortcut=False)  # 128(down) + 256(lateral) = 384 in
        
        # ===== Detection Head =====
        self.head = DetectionHead([c(64), c(128), c(256)], num_classes=num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 640, 640)
        
        Returns:
            Tensor of shape (B, 8400, 4 + 1 + num_classes)
            8400 = 80*80 + 40*40 + 20*20 anchor points
        """
        # Backbone
        x = self.stem(x)        # 320
        x = self.stage1(x)      # 160
        p3_backbone = self.stage2(x)      # 80x80 (P3)
        p4_backbone = self.stage3(p3_backbone)     # 40x40 (P4)
        p5_backbone = self.stage4(p4_backbone)   # 20x20 (P5)
        
        # FPN (top-down)
        # P5 -> P4
        p5_lat = self.lateral_p5(p5_backbone)  # 128 channels
        p5_up = self.upsample(p5_lat)          # 40x40
        p4_fused = self.fpn_p4(torch.cat([p5_up, p4_backbone], 1))  # 40x40
        
        # P4 -> P3
        p4_lat = self.lateral_p4(p4_fused)     # 64 channels
        p4_up = self.upsample(p4_lat)          # 80x80
        p3_fused = self.fpn_p3(torch.cat([p4_up, p3_backbone], 1))  # 80x80
        
        # PAN (bottom-up)
        # P3 -> P4
        p3_down = self.downsample_p3(p3_fused)       # 40x40
        n4 = self.pan_p4(torch.cat([p3_down, p4_fused], 1))  # 40x40
        
        # P4 -> P5
        n4_down = self.downsample_p4(n4)       # 20x20
        n5 = self.pan_p5(torch.cat([n4_down, p5_backbone], 1))  # 20x20
        
        # Detection
        features = [p3_fused, n4, n5]  # 80x80, 40x40, 20x20
        return self.head(features)
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_yolo_nano(num_classes=config.NUM_CLASSES, pretrained=False):
    """
    Factory function to create YOLOv11n model.
    
    Args:
        num_classes: Number of detection classes
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        YOLOv11n model instance
    """
    model = YOLOv11n(num_classes=num_classes)
    
    if pretrained:
        # TODO: Load pretrained weights from checkpoint
        pass
    
    return model


if __name__ == '__main__':
    # Test model
    model = YOLOv11n(num_classes=config.NUM_CLASSES)
    x = torch.randn(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
