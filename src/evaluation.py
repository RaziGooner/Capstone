import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy, Precision, Recall
from tqdm import tqdm
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_metrics(test_loader, model, threshold=0.5):
    model.eval()
    
    # Initialize metric objects from SMP
    iou_metric = IoU(threshold=threshold)  # Jaccard Index
    dice_metric = Fscore(threshold=threshold)  # Dice Score (F1 Score)
    accuracy_metric = Accuracy(threshold=threshold)
    precision_metric = Precision(threshold=threshold)
    recall_metric = Recall(threshold=threshold)
    
    # Lists to store per-batch results
    iou_scores = []
    dice_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Get model predictions
            logits = model(images)
            preds = torch.sigmoid(logits)  # Convert logits to probabilities
            
            # Compute metrics for the batch
            iou = iou_metric(preds, masks)
            dice = dice_metric(preds, masks)
            accuracy = accuracy_metric(preds, masks)
            precision = precision_metric(preds, masks)
            recall = recall_metric(preds, masks)
            
            # Store batch-wise results
            iou_scores.append(iou.item())
            dice_scores.append(dice.item())
            accuracy_scores.append(accuracy.item())
            precision_scores.append(precision.item())
            recall_scores.append(recall.item())
    
    # Calculate mean scores
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    
    # Formatted output
    print("\nModel Evaluation Metrics:")
    print("-" * 30)
    print(f"{'Metric':<12} {'Value':>10}")
    print("-" * 30)
    print(f"{'IoU':<12} {mean_iou:>10.4f}")
    print(f"{'Dice':<12} {mean_dice:>10.4f}")
    print(f"{'Accuracy':<12} {mean_accuracy:>10.4f}")
    print(f"{'Precision':<12} {mean_precision:>10.4f}")
    print(f"{'Recall':<12} {mean_recall:>10.4f}")
    print("-" * 30)
    
    # Return dictionary as before
    return {
        'IoU': mean_iou,
        'Dice': mean_dice,
        'Accuracy': mean_accuracy,
        'Precision': mean_precision,
        'Recall': mean_recall
    }

