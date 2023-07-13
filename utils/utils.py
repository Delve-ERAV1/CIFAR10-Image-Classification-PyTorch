import yaml
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from albumentations.pytorch import ToTensorV2
from dataset.dataset import Cifar10SearchDataset 

from utils.test import *
from utils.train import *
from torch_lr_finder import LRFinder




def get_stats(trainloader):
  """
  Args:
      trainloader (trainloader): Original data with no preprocessing
  Returns:
      mean: per channel mean
      std: per channel std
  """
  train_data = trainloader.dataset.data

  print('[Train]')
  print(' - Numpy Shape:', train_data.shape)
  print(' - Tensor Shape:', train_data.shape)
  print(' - min:', np.min(train_data))
  print(' - max:', np.max(train_data))

  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])


def get_loader(transform=None, train=True):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      loader: DataLoader Object
  """
  if transform:
    trainset = Cifar10SearchDataset(transform=transform)
  else:
    trainset = Cifar10SearchDataset(root="~/data/cifar10", train=train,
                                    download=True)
  loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                       shuffle=train, num_workers=2)
  return(loader)


def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): Original data with no preprocessing
      device (str): cuda/CPU
  """
  print(summary(model, input_size=(3, 32, 32)))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_device():
  """
  Returns:
      device (str): device type
  """
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # For reproducibility
  if cuda:
      torch.cuda.manual_seed(SEED)
  else:
    torch.manual_seed(SEED)

  return(device)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:denorm
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# functions to show an image
def imshow(img):
    img = (img)    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def process_config(file_name):
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print(" Loading Configuration File")
            return config
        except ValueError:
            print("Invalid file format")
            exit(-1)

def plot_metrics(metrics):
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25,6)
    train_accuracy,train_losses,test_accuracy,test_losses  = metrics
    
    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(test_losses), 'b', label="Validation Loss")
    
    # Label the plot.
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")
    
    # Label the plot.
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.show()


def LR_Finder(model, criterion, optimizer, trainloader):

  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, end_lr=1, num_iter=200)
  max_lr = lr_finder.plot(suggest_lr=True, skip_start=0, skip_end=0)
  lr_finder.reset()
  
  return(max_lr[1])


def train_model(model, device, trainloader, testloader, criterion, scheduler, optimizer, EPOCHS, metric):
  
  train_losses, train_acc = [], []
  test_losses, test_acc = [], []

  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)

      train(model, device, trainloader, epoch, criterion, scheduler, optimizer, [train_losses, train_acc])
      test(model, device, testloader, criterion, [test_losses, test_acc])