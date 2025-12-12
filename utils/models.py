import torch
import torch.nn as nn
import torch.nn.functional as F

# Import detection models
from utils.detection_models import YOLOv11n, create_yolo_nano, DetectionHead
from utils.detection_loss import DetectionLoss
from utils.detection_metrics import DetectionEvaluator

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for testing FedProx and DP integration.
    Designed for small inputs like CIFAR-10 or localized traffic patches (32x32).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
