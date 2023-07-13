import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_train_transform(mu, sigma):

    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    train_transform = A.Compose(
      [ 
          A.Normalize(mean=mu, std=sigma, always_apply=True),
          A.RandomCrop(height=32, width=32, padding=4, padding_mode='reflect', always_apply=True),
          A.HorizontalFlip(),
          A.Cutout(fill_value=mu),
          ToTensorV2(),
      ]
  )

    return(train_transform)



def get_test_transform(mu, sigma):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
        """
    test_transform = A.Compose([
                            A.Normalize(
                                mean=(mu),
                                std=(sigma)),
                            ToTensorV2(),
])
    return(test_transform)



def no_transform():
    return(A.Compose([A.Normalize()]))
