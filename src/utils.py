import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_sample(df, index=0):

    row = df.iloc[index]
    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

    print(f'Shape of Image: {image.shape} & Shape of Mask: {mask.shape}\n')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3))
    ax1.set_title('IMAGE')
    ax1.imshow(image)
    ax1.axis('off')

    ax2.set_title('MASK/ Label')
    ax2.imshow(mask, cmap='gray')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def train_fn(data_loader, model, optimizer ):
  model.train()
  total_loss =0.0
  for images, masks in tqdm(data_loader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits,loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(data_loader)

def eval_fn(data_loader, model):
  model.eval()
  total_loss =0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)
      logits,loss = model(images, masks)
      total_loss += loss.item()

  return total_loss / len(data_loader)

def plot_loss(losses, title='Training Loss', label='Training Loss', save_path=None):

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()