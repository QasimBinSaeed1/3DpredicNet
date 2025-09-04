import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# Define paths
# truth_folder = './/predict/output_/'
# predict_folder = './predict/output/'
truth_folder = './data/pothole600_real/val/depth'  # Ground truth depth maps
predict_folder = './yolo_result/'


# Transformation to convert PIL images to PyTorch tensors
to_tensor = transforms.ToTensor()

# Read and convert images to PyTorch tensors
def read_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale (single-channel)
    return to_tensor(img).long().squeeze(0)  # Remove the channel dimension if present

# Calculate metrics
def compute_metrics(true_labels, pred_labels, num_classes):
    # Flatten the tensors to compute metrics
    true_labels_flat = true_labels.flatten()
    pred_labels_flat = pred_labels.flatten()

    # Check if the shapes match
    assert true_labels_flat.shape == pred_labels_flat.shape, \
        f"Shape mismatch: true_labels_flat shape {true_labels_flat.shape}, pred_labels_flat shape {pred_labels_flat.shape}"
    
    # Compute Precision, Recall, F1 Score, and Accuracy
    precision, recall, f1 = [], [], []
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((true_labels_flat == i) & (pred_labels_flat == i)).sum().item()
        fp = ((true_labels_flat != i) & (pred_labels_flat == i)).sum().item()
        fn = ((true_labels_flat == i) & (pred_labels_flat != i)).sum().item()
        
        # Precision and recall for class i
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision.append(p)
        recall.append(r)
        
        # F1 Score for class i
        f1_i = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1.append(f1_i)
    
    # Compute overall accuracy
    accuracy = (true_labels_flat == pred_labels_flat).sum().item() / len(true_labels_flat)
    
    # Compute mIoU
    iou_list = []
    for i in range(num_classes):
        intersection = ((true_labels_flat == i) & (pred_labels_flat == i)).sum().item()
        union = ((true_labels_flat == i) | (pred_labels_flat == i)).sum().item()
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)
    
    mIoU = np.mean(iou_list)
    
    return np.mean(precision), np.mean(recall), accuracy, np.mean(f1), mIoU

# Main evaluation function
def evaluate_model(truth_folder, predict_folder):
    # Get the list of image filenames
    truth_images = sorted([f for f in os.listdir(truth_folder) if f.endswith('.png')])
    predict_images = sorted([f for f in os.listdir(predict_folder) if f.endswith('.png')])
    
    num_classes = len(np.unique(read_image(os.path.join(truth_folder, truth_images[0]))))  # Assumes all images have the same number of classes
    
    all_true_labels = []
    all_pred_labels = []
    
    start_time = time.time()  # Start timing
    
    for truth_image, predict_image in zip(truth_images, predict_images):
        true_labels = read_image(os.path.join(truth_folder, truth_image))
        pred_labels = read_image(os.path.join(predict_folder, predict_image))
        
        # Check image sizes
        assert true_labels.shape == pred_labels.shape, \
            f"Image size mismatch: true_labels shape {true_labels.shape}, pred_labels shape {pred_labels.shape}"
        
        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)
    
    end_time = time.time()  # End timing
    
    # Stack all images to create a single tensor
    all_true_labels = torch.cat(all_true_labels, dim=0)  # Concatenate along the batch dimension
    all_pred_labels = torch.cat(all_pred_labels, dim=0)
    
    # Compute metrics
    precision, recall, accuracy, f1, mIoU = compute_metrics(all_true_labels, all_pred_labels, num_classes)
    
    # Calculate FPS
    total_time = end_time - start_time
    num_images = len(truth_images)
    fps = num_images / total_time if total_time > 0 else float('inf')
    
    return precision, recall, accuracy, f1, mIoU, fps

# Run evaluation
precision, recall, accuracy, f1, mIoU, fps = evaluate_model(truth_folder, predict_folder)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'mIoU: {mIoU:.4f}')
print(f'FPS: {fps:.2f}')
