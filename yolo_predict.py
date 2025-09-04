import argparse
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import the YOLO11mDepth model (assumes yolo11m_depth.py is in the same directory)
from unet import YOLO11lDepth

def get_args():
    parser = argparse.ArgumentParser(description='Predict depth maps using YOLO11mDepth and save results')
    parser.add_argument('--model', type=str, 
                        default='yolocheckpoints/checkpoint_epoch300_depth.pth',
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--input-dir', type=str, default='data/yolo_data_440/val/rgb', 
                        help='Directory containing input RGB images')
    parser.add_argument('--output-dir', type=str, default='yolo_result', 
                        help='Directory to save predicted depth maps')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for inference (cuda or cpu)')
    parser.add_argument('--img-size', type=int, default=800, help='Input image size (should be 800 for YOLO11mDepth)')
    
    return parser.parse_args()

class DepthPredictionDataset(Dataset):
    def __init__(self, img_dir, img_size=800):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.img_paths = list(self.img_dir.glob('*.png')) + list(self.img_dir.glob('*.jpg'))
        if not self.img_paths:
            raise RuntimeError(f'No images found in {img_dir}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path.name

def predict_and_save(model, data_loader, output_dir, device):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch, img_names in tqdm(data_loader, desc='Predicting'):
            batch = batch.to(device)
            predictions = model(batch)  # Shape: [batch_size, 3, 800, 800]
            
            # Convert predictions to images
            predictions = predictions.permute(0, 2, 3, 1).cpu().numpy()  # Shape: [batch_size, 800, 800, 3]
            for pred, img_name in zip(predictions, img_names):
                # Normalize to [0, 255] and save as RGB image
                pred = (pred * 255).astype(np.uint8)
                pred_img = Image.fromarray(pred, mode='RGB')
                pred_img.save(output_dir / f'pred_{img_name}')

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    # Initialize and load model
    model = YOLO11lDepth(in_channels=3, nc=3)  # Set nc=3 to match checkpoint
    try:
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.model}')
    except Exception as e:
        logging.error(f'Failed to load model from {args.model}: {e}')
        raise

    model.to(device=device, memory_format=torch.channels_last)

    # Create dataset and dataloader
    try:
        dataset = DepthPredictionDataset(args.input_dir, img_size=args.img_size)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=os.cpu_count(), pin_memory=True)
        logging.info(f'Loaded {len(dataset)} images from {args.input_dir}')
    except Exception as e:
        logging.error(f'Failed to load dataset from {args.input_dir}: {e}')
        raise

    # Predict and save results
    predict_and_save(model, data_loader, args.output_dir, device)
    logging.info(f'Predictions saved to {args.output_dir}')

if __name__ == '__main__':
    main()