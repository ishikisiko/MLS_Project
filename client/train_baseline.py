import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.detection_models import YOLOv11n
from utils.data_loader import UADetracDataset
from utils.detection_loss import DetectionLoss
try:
    from client.training import train_detection_epoch, evaluate_detection
except ImportError:
    from training import train_detection_epoch, evaluate_detection

def download_dataset(api_key, data_path):
    """
    Download UA-DETRAC-10K-SAMPLE dataset from Roboflow.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Error: 'roboflow' package is not installed. Please install it using 'pip install roboflow'.")
        sys.exit(1)

    print(f"Downloading dataset to {data_path}...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("vehicle-detection-loakn").project("ua-detrac-10k-sample")
    # Using the latest version or a specific one. The user didn't specify version, but usually v1 or latest is good.
    # We'll try to get the latest version.
    version = project.version(1) 
    dataset = version.download("yolov8", location=data_path)
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location

def validate_and_fix_dataset_path(path):
    """
    Validates if the dataset path contains 'train/images'.
    If not, searches for it in immediate subdirectories.
    """
    # Check if path itself is valid
    expected_train = os.path.join(path, 'train', 'images')
    if os.path.exists(expected_train):
        # We don't check for file count strictly here as some splits might be small, 
        # but the directory must exist.
        return path
    
    # Check subdirectories
    if os.path.exists(path):
        print(f"Dataset not found directly in {path}. Checking subdirectories...")
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                candidate_train = os.path.join(item_path, 'train', 'images')
                if os.path.exists(candidate_train):
                    print(f"Found dataset in subdirectory: {item_path}")
                    return item_path
    
    print(f"Warning: Could not find valid dataset structure in {path}")
    # List contents to help debugging
    if os.path.exists(path):
        print(f"Contents of {path}: {os.listdir(path)}")
    return path

def main():
    parser = argparse.ArgumentParser(description="Train First Stage Baseline (YOLOv11n) on UA-DETRAC-10K-SAMPLE")
    parser.add_argument("--api-key", type=str, default="z6wNBWkCaVkCLEkgN8Y4", help="Roboflow API Key")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--data-dir", type=str, default=os.path.join(config.PROJECT_ROOT, "data", "ua-detrac-10k"), help="Directory to save dataset")
    
    args = parser.parse_args()

    # 1. Download Dataset
    train_images_dir = os.path.join(args.data_dir, 'train', 'images')
    # Check if data exists and is not empty
    data_exists = os.path.exists(args.data_dir) and \
                  os.path.exists(train_images_dir) and \
                  len(os.listdir(train_images_dir)) > 0

    if not data_exists:
        if os.path.exists(args.data_dir):
            print(f"Dataset directory {args.data_dir} exists but appears empty or incomplete. Removing to re-download...")
            import shutil
            shutil.rmtree(args.data_dir)
        
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"Downloading dataset using Roboflow API...")
        dataset_path = download_dataset(args.api_key, args.data_dir)
    else:
        print(f"Dataset directory {args.data_dir} already exists and contains data. Using existing data.")
        dataset_path = args.data_dir

    # Validate and fix path if necessary
    dataset_path = validate_and_fix_dataset_path(dataset_path)

    # 2. Setup DataLoaders
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)), 
        transforms.ToTensor(),
    ])

    # Roboflow yolov8 format usually has 'train', 'valid', 'test' folders
    train_dataset = UADetracDataset(root_dir=dataset_path, split='train', transform=transform)
    val_dataset = UADetracDataset(root_dir=dataset_path, split='valid', transform=transform)

    # Use more workers and pin_memory for faster data loading
    # Adjust num_workers based on CPU cores, but 8-16 is usually good for high-end GPUs
    num_workers = min(os.cpu_count(), 8) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=UADetracDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=UADetracDataset.collate_fn,
        num_workers=0, # Use 0 workers for validation to avoid hanging on Windows
        pin_memory=True,
        persistent_workers=False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Initialize Model
    print(f"Initializing YOLOv11n model on {args.device}...")
    model = YOLOv11n(num_classes=4).to(args.device) # 4 classes: bus, car, truck, van

    # 4. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = DetectionLoss(num_classes=4)

    # 5. Training Loop
    best_map = 0.0
    save_path = os.path.join(config.PROJECT_ROOT, "client", "baseline_model.pth")

    print("Starting training...")
    for epoch in range(args.epochs):
        # reuse existing training function
        # global_model_params is None because this is standalone training
        avg_loss_dict = train_detection_epoch(
            model=model,
            global_model_params=None,
            train_loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            loss_fn=loss_fn
        )

        # Log progress
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {avg_loss_dict['total_loss']:.4f} "
              f"(Box: {avg_loss_dict['box_loss']:.4f}, "
              f"Obj: {avg_loss_dict['obj_loss']:.4f}, "
              f"Cls: {avg_loss_dict['cls_loss']:.4f})")

        # Evaluate on validation set
        print("Evaluating on validation set...")
        # Use a lower confidence threshold for mAP calculation to capture the full PR curve
        metrics = evaluate_detection(model, val_loader, args.device, num_classes=4, conf_threshold=0.001)
        
        print(f"Validation mAP@0.5: {metrics['mAP@0.5']:.4f}")
        print(f"Validation mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        
        # Save best model based on mAP@0.5
        current_map = metrics['mAP@0.5']
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (mAP@0.5: {best_map:.4f})")


    print("Training complete.")

if __name__ == "__main__":
    main()
