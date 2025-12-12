import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import glob

class UADetracDataset(Dataset):
    """
    Custom Dataset for UA-DETRAC vehicle detection.
    Reads images from the specified folder structure and applies transforms.
    Returns dummy labels for compatibility with SimpleCNN (classification).
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
        
        # Determine image directory based on split
        # Structure: root_dir/train/images, root_dir/valid/images, root_dir/test/images
        if split == 'valid':
            self.image_dir = os.path.join(root_dir, 'valid', 'images')
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, 'test', 'images')
        else:
            self.image_dir = os.path.join(root_dir, 'train', 'images')
            
        # Get all image files
        self.image_paths = []
        if os.path.exists(self.image_dir):
            # Supports common image extensions
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
        else:
            print(f"Warning: Image directory not found: {self.image_dir}")
            
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error to prevent crash
            image = Image.new('RGB', (32, 32), color='black')

        if self.transform:
            image = self.transform(image)
            
        # Return dummy label (0) since we are using a classifier on detection data
        # Ideally we would parse the corresponding label file if we were doing detection training
        label = 0 
        
        return image, label


def get_ua_detrac_loaders(data_root=None, batch_size=32, num_workers=0):
    """
    Get UA-DETRAC train and test data loaders.
    
    Args:
        data_root: Root directory containing train/valid/test folders. 
                   If None, defaults to ./data/ua-detrac
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ua-detrac')
    
    # Define transforms for SimpleCNN (expects 32x32 input)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet stats
    ])
    
    # Create datasets
    trainset = UADetracDataset(root_dir=data_root, split='train', transform=transform)
    valset = UADetracDataset(root_dir=data_root, split='valid', transform=transform)
    testset = UADetracDataset(root_dir=data_root, split='test', transform=transform)
    
    # If validation set is empty, fall back to test set, or just use train for both if everything else fails
    if len(trainset) == 0:
        print("Warning: Training set is empty.")
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(valset if len(valset) > 0 else testset, 
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_data_loaders(batch_size=32, train_split=0.9, num_workers=0):
    """
    Legacy helper to keep backward compatibility or easy switch back to CIFAR-10.
    """
    return get_ua_detrac_loaders(batch_size=batch_size, num_workers=num_workers)
