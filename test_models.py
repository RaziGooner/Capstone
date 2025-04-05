import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from aug_torchdataset import SegmentationDataset, get_test_augs
from model import SegmentationModel
import pandas as pd
impoal setup)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256
BATCisualization
NUM to visualize
base_dir = "Capstone"
test_csv = f"./{base_dir}/test_data.csv"
model_path_baseline = f"./{base_dir}/model/baseline_model.pt"
model_path_al = f"./{base_dir}/model/al_model.pt"

# Load test data
test_df = pd.read_csv(test_csv)
testset = SegmentationDataset(test_df, get_test_augs())
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Load models
baseline_model = SegmentationModel()
baseline_model.load_state_dict(torch.load(model_path_baseline))
baseline_model.to(DEVICE)
baseline_model.eval()

al_model = SegmentationModel()
al_model.load_state_dict(torch.load(model_path_al))
al_mot tensor to numpy for visualization
def tensor_to_numpy(tensor):
    tensor = tensor.cpu().numpy().squeeze()
    if tensor.ndiin [1, 3]:  # Handle channel dimension
        tensor = tensor.transtensor.max() > 1:  # Normalize if needed
        tensor = tensor / tensor.max()
    return tensor

# Visualization function
def visualize_prediction(image, pred, gt, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tensor_to_numpy(image))
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(tensor_to_numpy(pred), cmap='gray')
    axes[1].set_title(f"Prediction ({title})")
    axes[1].axis('off')
    
    axes[2].imshow(tensor_to_numpy(gt), cmap='gray')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test and visualize
print(f"Visualizing {NUM_SAMPLES} samples from the test set...")
with torch.no_grad():
    for i, (images, masks) in enumerate(testloader):
        if i >= NUM_SAMPLES:
            break
        EVICE), masks.to(DEVICE)
        
        # Baseline model prediction
        baseline_presinstance(baseline_pred, tuple):  # Handle case where model returns tuple
            baseline_pred    baseliApply sigmoid if logits
        
        # Active learning model prediction
        al_pred = al_model(images)
        if isinstance(al_pred, tuple):
  al_pred = torch.sigmoid(al_pred)
        
        # Move to CPU for visualization
        image, mask = imagesred, al_pred = baseline_pred[0], al_pred[0]
        
        # Visualize for baseline
        print(f"\nSample {i+1} - Baseline Mode, baseline_pred, mask, "Baseline Model")
        
        # Visualize for active learning
        print(f"Sample {i+1} - Active Learning Model")
        visualize_prediction(image, al_pred, mask, "Active Learning Model")

if __name__ == "__main__":
    print("Testing complete.")