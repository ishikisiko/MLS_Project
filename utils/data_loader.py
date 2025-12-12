import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
from utils import config

# Class names mapping
NAMES = config.CLASS_NAMES

class UADetracDataset(Dataset):
    """
    Custom Dataset for UA-DETRAC vehicle detection.
    Reads images and corresponding label files (YOLO format).
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'valid', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Determine directories
        # Structure: root_dir/split/images and root_dir/split/labels
        if split == 'valid':
            self.image_dir = os.path.join(root_dir, 'valid', 'images')
            self.label_dir = os.path.join(root_dir, 'valid', 'labels')
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, 'test', 'images')
            self.label_dir = os.path.join(root_dir, 'test', 'labels')
        else:
            self.image_dir = os.path.join(root_dir, 'train', 'images')
            self.label_dir = os.path.join(root_dir, 'train', 'labels')
            
        # Get all image files
        self.image_paths = []
        if os.path.exists(self.image_dir):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            self.image_paths = sorted(self.image_paths) # Ensure consistent order
        else:
            print(f"Warning: Image directory not found: {self.image_dir}")
            
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (config.INPUT_SIZE, config.INPUT_SIZE), color='black')
            w, h = config.INPUT_SIZE, config.INPUT_SIZE

        # Load Label
        # Assumes label file has same basename as image but .txt extension
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Format: class x_center y_center width height
                    l = line.strip().split()
                    if len(l) == 5:
                        cls = int(l[0])
                        # Basic validation for class index
                        if 0 <= cls < len(NAMES):
                            labels.append([cls, float(l[1]), float(l[2]), float(l[3]), float(l[4])])
        
        # Convert to tensor
        nL = len(labels)
        if nL:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            
        return image, labels_tensor, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to handle variable number of objects per image.
        Adds a batch index to the targets.
        """
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class MockDetectionDataset(Dataset):
    """
    Mock Dataset for testing object detection without real data.
    Generates random images and valid YOLO-format labels.
    """
    def __init__(self, size=100, transform=None, num_classes=None):
        """
        Args:
            size (int): Number of samples in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_classes (int): Number of classes for detection.
        """
        self.size = size
        self.transform = transform
        self.num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        self.min_objects, self.max_objects = config.MOCK_NUM_OBJECTS_RANGE
        
        # Pre-generate random seeds for reproducibility within the same run
        self.seeds = np.random.randint(0, 1000000, size=size)
        
        print(f"Created MockDetectionDataset with {size} samples")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Use seed for this index to ensure consistent data per index
        np.random.seed(self.seeds[idx])
        
        # Generate random image (as PIL Image first if transform expects it)
        img_array = np.random.randint(0, 256, (config.INPUT_SIZE, config.INPUT_SIZE, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, 'RGB')
        
        h, w = config.INPUT_SIZE, config.INPUT_SIZE
        
        # Generate random number of objects
        num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        # Generate valid YOLO labels
        labels = []
        for _ in range(num_objects):
            cls = np.random.randint(0, self.num_classes)
            # Generate valid bounding box (x_center, y_center, width, height) in [0, 1]
            # Ensure reasonable box sizes
            box_w = np.random.uniform(0.05, 0.4)
            box_h = np.random.uniform(0.05, 0.4)
            x_center = np.random.uniform(box_w / 2, 1 - box_w / 2)
            y_center = np.random.uniform(box_h / 2, 1 - box_h / 2)
            labels.append([cls, x_center, y_center, box_w, box_h])
        
        # Convert to tensor
        if labels:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Create mock path for compatibility
        mock_path = f"mock_image_{idx:06d}.jpg"
        
        return image, labels_tensor, mock_path, (h, w)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to handle variable number of objects per image.
        Same as UADetracDataset.collate_fn for compatibility.
        """
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def get_mock_loaders(batch_size=32, num_workers=0):
    """
    Get mock train and test data loaders for object detection testing.
    """
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    
    trainset = MockDetectionDataset(
        size=config.MOCK_TRAIN_SIZE,
        transform=transform,
        num_classes=config.NUM_CLASSES
    )
    valset = MockDetectionDataset(
        size=config.MOCK_VAL_SIZE,
        transform=transform,
        num_classes=config.NUM_CLASSES
    )
    testset = MockDetectionDataset(
        size=config.MOCK_TEST_SIZE,
        transform=transform,
        num_classes=config.NUM_CLASSES
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=MockDetectionDataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        valset if len(valset) > 0 else testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=MockDetectionDataset.collate_fn
    )
    
    return train_loader, val_loader


def get_ua_detrac_loaders(data_root=None, batch_size=32, num_workers=0, use_mock=None):
    """
    Get UA-DETRAC train and test data loaders for Object Detection.
    
    Args:
        data_root: Root directory of the dataset (ignored if using mock data)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        use_mock: Override config.USE_MOCK_DATA. If None, uses config value.
    
    Returns:
        train_loader, test_loader tuple
    """
    # Determine if using mock data
    if use_mock is None:
        use_mock = config.USE_MOCK_DATA
    
    if use_mock:
        print("Using mock (simulated) dataset for testing...")
        return get_mock_loaders(batch_size=batch_size, num_workers=num_workers)
    
    # Real data path
    if data_root is None:
        data_root = config.DEFAULT_DATA_ROOT
    
    print(f"Using real UA-DETRAC dataset from: {data_root}")
    
    # Standard YOLO-style resizing (e.g., to 640x640)
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)), 
        transforms.ToTensor(),
        # transforms.Normalize(...) # Often YOLO doesn't require explicit existing normalization if model handles it, but keeping it standard is okay.
        # For simplicity in this step, omitting complex augmentations.
    ])
    
    # Create datasets
    trainset = UADetracDataset(root_dir=data_root, split='train', transform=transform)
    valset = UADetracDataset(root_dir=data_root, split='valid', transform=transform)
    testset = UADetracDataset(root_dir=data_root, split='test', transform=transform)
    
    if len(trainset) == 0:
        print("Warning: Training set is empty.")
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers,
                                               collate_fn=UADetracDataset.collate_fn)
    
    test_loader = torch.utils.data.DataLoader(valset if len(valset) > 0 else testset, 
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              collate_fn=UADetracDataset.collate_fn)
    
    return train_loader, test_loader


def get_data_loaders(batch_size=32, train_split=0.9, num_workers=0):
    """
    Legacy helper.
    """
    return get_ua_detrac_loaders(batch_size=batch_size, num_workers=num_workers)
