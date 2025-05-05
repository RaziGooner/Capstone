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

def expected_model_change(model, unlabeled_loader, budget_per_iter, device='cuda'):
    model.eval()
    gradient_norms = []
    indices = []
    images_list = []  
    
    for idx, (images, _) in enumerate(tqdm(unlabeled_loader, desc="Computing expected model change")):
        images = images.to(device).requires_grad_(True)
        logits = model(images)
        # Use entropy as a proxy for loss (consistent with your uncertainty sampling)
        probs = torch.sigmoid(logits)
        entropy = -torch.sum(torch.cat([1 - probs, probs], dim=1) * 
                            torch.log(torch.cat([1 - probs, probs], dim=1) + 1e-10), dim=1)
        loss = entropy.mean()
        
        model.zero_grad()
        loss.backward()
        
        # Compute gradient norm for each sample in the batch
        grad_norm = torch.norm(images.grad.view(images.size(0), -1), dim=1).cpu().numpy()
        gradient_norms.extend(grad_norm)
        indices.extend(range(idx * BATCH_SIZE, idx * BATCH_SIZE + images.size(0)))
        images_list.append(images.detach().cpu())  # Store for potential visualization
    
    gradient_norms = np.array(gradient_norms)
    indices = np.array(indices)
    sorted_indices = np.argsort(gradient_norms)[::-1]  # Sort in descending order
    selected_indices = sorted_indices[:budget_per_iter]
    
    return [indices[i] for i in selected_indices], images_list, gradient_norms

def active_learning_loop(model, labeled_df, unlabeled_df, 
                        budget_per_iter=10, max_iterations=5, n_show=5, 
                        epochs_per_iter=5, strategy='uncertainty', 
                        uncertainty_mode='mean', topk_k=500, device='cuda'):
    current_labeled_df = labeled_df.copy()
    unlabeled_pool = SegmentationDataset(unlabeled_df, get_valid_augs())
    unlabeled_loader = DataLoader(unlabeled_pool, batch_size=BATCH_SIZE, shuffle=False)
    
    for iteration in range(max_iterations):
        print(f"\nActive Learning Iteration {iteration + 1}/{max_iterations}")
        
        if strategy == 'uncertainty':
            uncertainties = []
            indices = []
            images_list = []
            
            with torch.no_grad():
                for idx, (images, _) in enumerate(tqdm(unlabeled_loader, desc="Computing uncertainties")):
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    probs = torch.cat([1 - probs, probs], dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [batch_size, H, W]
                    
                    # Compute uncertainty score based on mode
                    if uncertainty_mode == 'mean':
                        score = entropy.mean(dim=[1, 2])
                    elif uncertainty_mode == 'max':
                        score = entropy.amax(dim=[1, 2])
                    elif uncertainty_mode == 'topk':
                        topk_vals, _ = torch.topk(entropy.view(entropy.shape[0], -1), k=topk_k, dim=1)
                        score = topk_vals.mean(dim=1)
                    else:
                        raise ValueError(f"Unsupported uncertainty_mode: {uncertainty_mode}. Use 'mean', 'max', or 'topk'.")
                    
                    uncertainties.extend(score.cpu().numpy())
                    indices.extend(range(idx * BATCH_SIZE, idx * BATCH_SIZE + images.size(0)))
                    images_list.append(images.cpu())
            
            uncertainties = np.array(uncertainties)
            indices = np.array(indices)
            sorted_indices = np.argsort(uncertainties)[::-1]
            selected_indices = [indices[i] for i in sorted_indices[:budget_per_iter]]
            scores = uncertainties  # For visualization
            
        elif strategy == 'model_change':
            selected_indices, images_list, scores = expected_model_change(
                model, unlabeled_loader, budget_per_iter, device=device
            )
            sorted_indices = np.argsort(scores)[::-1]  # Already sorted in function
        
        else:
            raise ValueError(f"Unsupported strategy: {strategy}. Use 'uncertainty' or 'model_change'.")
        
        # Visualization of top uncertain/changing samples
        print(f"\nDisplaying top {min(n_show, len(scores))} samples with highest {'uncertainty' if strategy == 'uncertainty' else 'gradient norm'} ({uncertainty_mode if strategy == 'uncertainty' else ''}):")
        for i in range(min(n_show, len(scores))):
            sel_idx = sorted_indices[i]
            batch_idx = sel_idx // BATCH_SIZE
            img_idx = sel_idx % BATCH_SIZE
            image = images_list[batch_idx][img_idx:img_idx+1].to(device)
            heatmap = get_gradcam(model, image, image)  # Assuming target_layer is input for simplicity
            heatmap_np = heatmap.squeeze(0).detach().cpu().numpy()
            img_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
            img_np = img_np.astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_np, (IMAGE_SIZE, IMAGE_SIZE))
            plt.figure(figsize=(6, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            im = plt.imshow(heatmap_resized, cmap='jet')
            plt.title(f"({'Uncertainty' if strategy == 'uncertainty' else 'Gradient Norm'}: {scores[sel_idx]:.4f})")
            plt.axis('off')
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Contribution', rotation=270, labelpad=15)
            plt.tight_layout()
            plt.show()
        
        # Update labeled and unlabeled datasets
        print(f"Selected {len(selected_indices)} samples for labeling.")
        new_labeled_df = unlabeled_df.iloc[selected_indices]
        current_labeled_df = pd.concat([current_labeled_df, new_labeled_df])
        unlabeled_df = unlabeled_df.drop(unlabeled_df.index[selected_indices]).reset_index(drop=True)
        unlabeled_pool = SegmentationDataset(unlabeled_df, get_valid_augs())
        unlabeled_loader = DataLoader(unlabeled_pool, batch_size=BATCH_SIZE, shuffle=False)
        
        # Retrain model
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