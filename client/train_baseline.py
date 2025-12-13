import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.detection_models import YOLOv11n
from utils.data_loader import UADetracDataset
from utils.detection_loss import DetectionLoss
try:
    from client.training import train_detection_epoch
except ImportError:
    from training import train_detection_epoch

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
    return dataset.location

def main():
    parser = argparse.ArgumentParser(description="Train First Stage Baseline (YOLOv11n) on UA-DETRAC-10K-SAMPLE")
    parser.add_argument("--api-key", type=str, default="TwdK954qNo", help="Roboflow API Key")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--data-dir", type=str, default=os.path.join(config.PROJECT_ROOT, "data", "ua-detrac-10k"), help="Directory to save dataset")
    
    args = parser.parse_args()

    # 1. Download Dataset
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"Downloading dataset using Roboflow API...")
        dataset_path = download_dataset(args.api_key, args.data_dir)
    else:
        print(f"Dataset directory {args.data_dir} already exists. Using existing data.")
        dataset_path = args.data_dir

    # 2. Setup DataLoaders
    # Roboflow yolov8 format usually has 'train', 'valid', 'test' folders
    train_dataset = UADetracDataset(root_dir=dataset_path, split='train')
    val_dataset = UADetracDataset(root_dir=dataset_path, split='valid')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=UADetracDataset.collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=UADetracDataset.collate_fn,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Initialize Model
    print(f"Initializing YOLOv11n model on {args.device}...")
    model = YOLOv11n(num_classes=4).to(args.device) # 4 classes: bus, car, truck, van

    # 4. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = DetectionLoss(num_classes=4)

    # 5. Training Loop
    best_loss = float('inf')
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

        # Simple validation (loss-based)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    images, targets, _, _ = batch_data
                else:
                    images, targets = batch_data
                
                images = images.to(args.device)
                targets = targets.to(args.device)
                
                preds = model(images)
                
                # Build targets for loss calculation
                obj_t, box_t, cls_t, mask = loss_fn.build_targets(preds, targets)
                
                # Calculate loss components (simplified for validation logging)
                # Note: This requires manually calling loss parts or reusing loss_fn if it supports eval
                # For now, we just track training loss or save every epoch
        
        # Save best model based on training loss (or implement proper validation loss)
        if avg_loss_dict['total_loss'] < best_loss:
            best_loss = avg_loss_dict['total_loss']
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
