import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.detection_models import YOLOv11n
from utils.data_loader import UADetracDataset, MockDetectionDataset
try:
    from client.training import evaluate_detection
except ImportError:
    from training import evaluate_detection

def validate_and_fix_dataset_path(path):
    """
    Validates if the dataset path contains 'train/images'.
    If not, searches for it in immediate subdirectories.
    """
    # Check if path itself is valid
    expected_valid = os.path.join(path, 'valid', 'images')
    if os.path.exists(expected_valid):
        return path
    
    # Check subdirectories
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                candidate_valid = os.path.join(item_path, 'valid', 'images')
                if os.path.exists(candidate_valid):
                    print(f"Found dataset in subdirectory: {item_path}")
                    return item_path
    
    print(f"Warning: Could not find valid dataset structure in {path}")
    return path

def main():
    parser = argparse.ArgumentParser(description="Verify mAP of Baseline or Compressed Models")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default=os.path.join(config.PROJECT_ROOT, "client", "baseline_model.pth"), 
                        help="Path to the model checkpoint")
    parser.add_argument("--width-mult", type=float, default=1.0, 
                        help="Width multiplier for the model (default: 1.0 for baseline, 0.5 for distilled student)")
    
    # Dataset arguments
    parser.add_argument("--data-dir", type=str, default=os.path.join(config.PROJECT_ROOT, "data", "ua-detrac-10k"), 
                        help="Directory of the dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    # Execution arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--conf-threshold", type=float, default=0.01, help="Confidence threshold for evaluation")
    parser.add_argument("--use-mock", action="store_true", help="Use mock dataset for testing purposes")
    
    # Compressed model flag (logic wrapper)
    parser.add_argument("--compressed", action="store_true", help="Verify compressed model (sets defaults for compressed model if path not provided)")
    
    args = parser.parse_args()

    # Handle --compressed flag convenience
    if args.compressed:
        print("Running in Compressed Model Verification Mode...")
        # If user didn't specify a custom path, assume a default name for compressed model? 
        # But we don't have a standard one. So we'll just rely on them providing the path or use the flag to adjust other params.
        # For example, if they say --compressed but keep default model path, we might warn them.
        
        # If the user explicitly asks for compressed but uses default baseline path, 
        # we might look for 'distilled_model.pth' or similar if it existed, but it doesn't by default.
        # So we'll just respect the provided path.
        
        # However, we can default width-mult to 0.5 if it's still 1.0 and compressed is set?
        # Let's not make too many magic assumptions, but printing a hint is good.
        if args.width_mult == 1.0:
            print("Note: You specified --compressed but --width-mult is 1.0. "
                  "If you are verifying a distilled student model, you might need --width-mult 0.5")

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first or provide the correct path.")
        sys.exit(1)

    # 1. Setup DataLoaders
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)), 
        transforms.ToTensor(),
    ])

    if args.use_mock:
        print("Using Mock Dataset...")
        val_dataset = MockDetectionDataset(size=50, transform=transform)
        collate_fn = MockDetectionDataset.collate_fn
    else:
        # Validate and fix path if necessary
        dataset_path = validate_and_fix_dataset_path(args.data_dir)
        
        if not os.path.exists(os.path.join(dataset_path, 'valid')):
             print(f"Error: Validation set not found in {dataset_path}")
             sys.exit(1)

        val_dataset = UADetracDataset(root_dir=dataset_path, split='valid', transform=transform)
        collate_fn = UADetracDataset.collate_fn
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0, # Windows friendly
        pin_memory=True
    )

    print(f"Validation samples: {len(val_dataset)}")

    # 2. Initialize Model
    print(f"Initializing YOLOv11n model (width_mult={args.width_mult}) on {args.device}...")
    model = YOLOv11n(num_classes=4, width_mult=args.width_mult).to(args.device)
    
    # Load weights
    print(f"Loading weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # 3. Evaluate
    print("Evaluating on validation set...")
    print(f"Confidence Threshold: {args.conf_threshold}")
    
    metrics = evaluate_detection(model, val_loader, args.device, num_classes=4, conf_threshold=args.conf_threshold)
    
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"mAP@0.5:      {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
