import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'ua-detrac')

# Model Configuration
MODEL_TYPE = 'detection'  # Options: 'classification', 'detection'
INPUT_SIZE = 640
NUM_CLASSES = 4
CLASS_NAMES = ['bus', 'car', 'truck', 'van']

# Federated Learning Configuration
MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50MB
DEFAULT_ROUNDS = 3
MIN_FIT_CLIENTS = 2
ROUND_TIMEOUT = 30.0

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Federated Learning - FedProx
FEDPROX_MU = 0.1

# Detection specific
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Loss Weights
BOX_WEIGHT = 7.5
OBJ_WEIGHT = 1.0
CLS_WEIGHT = 0.5

# Focal Loss
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Dataset Configuration
USE_MOCK_DATA = False # Set to True to use simulated data for testing
MOCK_TRAIN_SIZE = 100  # Number of training samples in mock dataset
MOCK_VAL_SIZE = 20     # Number of validation samples in mock dataset
MOCK_TEST_SIZE = 20    # Number of test samples in mock dataset
MOCK_NUM_OBJECTS_RANGE = (1, 5)  # Min and max objects per image in mock data
