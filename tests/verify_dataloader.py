import torch
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_loader

class TestUADetracLoader(unittest.TestCase):
    def setUp(self):
        # Create dummy directories if not exist (using current dir as root for test if needed)
        # But we will mock first to test logic without files
        pass

    @patch('utils.data_loader.os.path.exists')
    @patch('utils.data_loader.glob.glob')
    @patch('utils.data_loader.Image.open')
    @patch('builtins.open')
    def test_dataloader_output_shape(self, mock_open, mock_image_open, mock_glob, mock_exists):
        """Test dataset items and collate_fn with mocked data"""
        
        # Setup mocks
        mock_exists.return_value = True
        mock_glob.return_value = ['root/train/images/img1.jpg', 'root/train/images/img2.jpg']
        
        # Mock Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (800, 600) # w, h
        mock_image_open.return_value = mock_img
        
        # Mock Label File Reading
        # File 1: 2 objects
        # File 2: 1 object
        mock_file1 = ["0 0.5 0.5 0.2 0.2\n", "1 0.3 0.3 0.1 0.1\n"]
        mock_file2 = ["2 0.8 0.8 0.1 0.5\n"]
        
        # Configure mock_open to return different file contents
        mock_open.side_effect = [
            MagicMock(  # File 1 context manager
                __enter__=MagicMock(return_value=iter(mock_file1)),
                __exit__=MagicMock()
            ),
            MagicMock(  # File 2 context manager
                __enter__=MagicMock(return_value=iter(mock_file2)),
                __exit__=MagicMock()
            )
        ]

        # Init Dataset
        dataset = data_loader.UADetracDataset(root_dir='root', split='train', 
                                              transform=data_loader.transforms.Compose([
                                                  data_loader.transforms.Resize((640, 640)),
                                                  data_loader.transforms.ToTensor()
                                              ]))
        
        # Test __getitem__
        img, label, path, shape = dataset[0]
        self.assertEqual(img.shape, (3, 640, 640)) # Check resize
        self.assertEqual(label.shape, (2, 5)) # 2 objects, 5 coords (cls, x, y, w, h)
        self.assertEqual(label[0, 0], 0) # class 0
        self.assertEqual(label[1, 0], 1) # class 1
        
        # Test collate_fn with DataLoader
        # We need to manually call collate or use DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=data_loader.UADetracDataset.collate_fn)
        
        batch = next(iter(loader))
        imgs, targets, paths, shapes = batch
        
        self.assertEqual(imgs.shape, (2, 3, 640, 640))
        # Total targets: 2 from img1 + 1 from img2 = 3
        # Validates that targets are concatenated
        self.assertEqual(targets.shape, (3, 6)) 
        # Targets format: [batch_idx, cls, x, y, w, h]
        
        # Check batch indices
        self.assertEqual(targets[0, 0], 0) # First object belongs to batch 0
        self.assertEqual(targets[1, 0], 0) # Second object belongs to batch 0
        self.assertEqual(targets[2, 0], 1) # Third object belongs to batch 1
        
        print("\nVerification Successful: shapes and collation logic are correct.")

if __name__ == '__main__':
    unittest.main()
