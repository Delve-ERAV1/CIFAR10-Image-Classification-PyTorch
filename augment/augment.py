import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_transforms(means, stds):
  train_transforms = A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
          A.RandomCrop(height=32, width=32, always_apply=True),
          A.HorizontalFlip(),
          A.Cutout (fill_value=means),
          ToTensorV2(),
      ]
  )

  test_transforms = A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          ToTensorV2(),
      ]
  )

  return(train_transforms, test_transforms)
