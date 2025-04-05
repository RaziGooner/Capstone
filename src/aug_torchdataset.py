import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


## Augmentation for training and resizing for validation and tesg
def get_train_augs(IMAGE_SIZE=256):
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                     ])
def get_valid_augs(IMAGE_SIZE=256):
  return A.Compose([
                    A.Resize(IMAGE_SIZE,IMAGE_SIZE),             
                   ])
def get_test_augs(IMAGE_SIZE=256):
  return A.Compose([
                    A.Resize(IMAGE_SIZE,IMAGE_SIZE), 
                   ])

## Create pytorch dataset
class SegmentationDataset(Dataset):
  def __init__(self, df, augmentation):
    self.df = df
    self.augmentation=augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image_path = row.images
    mask_path = row.masks
    image =cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask,axis=-1)

    if self.augmentation:
      data = self.augmentation(image=image, mask=mask)
      image = data['image']
      mask = data['mask']

      image = np.transpose(image, (2,0,1)).astype(np.float32)
      mask = np.transpose(mask, (2,0,1)).astype(np.float32)

      image = torch.Tensor(image) / 255.0
      mask = torch.round(torch.Tensor(mask) / 255.0)

      return image, mask