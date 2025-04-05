from aug_torchdataset import SegmentationDataset, get_train_augs, get_test_augs, get_valid_augs
from utils import train_fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.001
IMAGE_SIZE = 256
BATCH_SIZE = 16


def get_gradcam(model, images, target_layer):
    model.eval()
    images = images.requires_grad_(True)
    logits = model(images)
    
    probs = torch.sigmoid(logits)
    probs = torch.cat([1 - probs, probs], dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    target = entropy.mean()
    
    model.zero_grad()
    target.backward()
    
    gradients = images.grad
    activations = target_layer
    
    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
    heatmap = torch.mean(activations * pooled_gradients, dim=1)
    heatmap = nn.functional.relu(heatmap)
    
    max_values, _ = torch.max(heatmap, dim=1, keepdim=True)
    max_values, _ = torch.max(max_values, dim=2, keepdim=True)
    heatmap /= (max_values + 1e-10)
    
    return heatmap

def active_learning_loop(model, labeled_df, unlabeled_df, 
                        budget_per_iter=10, max_iterations=5, n_show=5, epochs_per_iter=5):
    current_labeled_df = labeled_df.copy()
    unlabeled_pool = SegmentationDataset(unlabeled_df, get_valid_augs())
    
    for iteration in range(max_iterations):
        print(f"\nActive Learning Iteration {iteration + 1}/{max_iterations}")
        
        # Uncertainty computation and visualization
        model.eval()
        uncertainties = []
        indices = []
        images_list = []
        
        unlabeled_loader = DataLoader(unlabeled_pool, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for idx, (images, _) in enumerate(tqdm(unlabeled_loader, desc="Computing uncertainties")):
                images = images.to(DEVICE)
                logits = model(images)
                entropy = -torch.sum(torch.cat([1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1) * 
                                    torch.log(torch.cat([1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1) + 1e-10), dim=1)
                mean_entropy = entropy.mean(dim=[1, 2])
                uncertainties.extend(mean_entropy.cpu().numpy())
                indices.extend(range(idx * BATCH_SIZE, idx * BATCH_SIZE + images.size(0)))
                images_list.append(images.cpu())
        
        uncertainties = np.array(uncertainties)
        indices = np.array(indices)
        sorted_indices = np.argsort(uncertainties)[::-1]
        
        print(f"\nDisplaying top {min(n_show, len(uncertainties))} samples with highest uncertainty:")
        for i in range(min(n_show, len(uncertainties))):
            sel_idx = sorted_indices[i]
            batch_idx = sel_idx // BATCH_SIZE
            img_idx = sel_idx % BATCH_SIZE
            image = images_list[batch_idx][img_idx:img_idx+1].to(DEVICE)
            heatmap = get_gradcam(model, image, image)
            heatmap_np = heatmap.squeeze(0).detach().cpu().numpy()
            img_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
            img_np = img_np.astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_np, (IMAGE_SIZE, IMAGE_SIZE))
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            im = plt.imshow(heatmap_resized, cmap='jet')
            plt.title(f"Grad-CAM (Uncertainty: {uncertainties[sel_idx]:.4f})")
            plt.axis('off')
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Contribution to Uncertainty', rotation=270, labelpad=15)
            plt.tight_layout()
            plt.show()
        
        selected_indices = sorted_indices[:budget_per_iter]
        selected_df_indices = [indices[i] for i in selected_indices]
        print(f"Selected {len(selected_df_indices)} samples for labeling.")
        
        new_labeled_df = unlabeled_df.iloc[selected_df_indices]
        current_labeled_df = pd.concat([current_labeled_df, new_labeled_df])
        unlabeled_df = unlabeled_df.drop(unlabeled_df.index[selected_df_indices]).reset_index(drop=True)
        unlabeled_pool = SegmentationDataset(unlabeled_df, get_valid_augs())
        
        # Retrain model for multiple epochs
        trainset = SegmentationDataset(current_labeled_df, get_train_augs())
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        for epoch in range(epochs_per_iter):
            train_loss = train_fn(trainloader, model, optimizer)
            print(f"Iteration {iteration+1}, Epoch {epoch+1}/{epochs_per_iter} - Train Loss: {train_loss:.4f}")
        
        if len(unlabeled_df) < budget_per_iter:
            print("Unlabeled pool exhausted.")
            break
    
    return model

# Visualization function for explainability
def visualize_explanation(image, mask, prediction, entropy_map, gradcam_map, idx, save_path='./Capstone/explainability'):
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(mask[0].cpu().numpy(), cmap='gray')
    axes[1].set_title("Ground Truth (if available)")
    axes[2].imshow(torch.sigmoid(prediction[0]).cpu().numpy(), cmap='gray')
    axes[2].set_title("Prediction")
    axes[3].imshow(entropy_map.cpu().numpy(), cmap='hot')
    axes[3].set_title("Entropy Map")
    axes[4].imshow(gradcam_map, cmap='jet', alpha=0.5)
    axes[4].imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray', alpha=0.5)
    axes[4].set_title("Grad-CAM Overlay")
    
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Sample {idx} Explainability")
    plt.savefig(f"{save_path}/sample_{idx}.png")
    plt.close()