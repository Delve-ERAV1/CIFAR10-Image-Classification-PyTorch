import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel, stride=1):
    super(ResBlock, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.BatchNorm2d(in_channel), 
        nn.ReLU(),

        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.BatchNorm2d(out_channel), 
        nn.ReLU(),
    )

  def forward(self, x):
    return(self.conv(x))


class ResNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ResNet, self).__init__()

    self.prep_layer = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.BatchNorm2d(64), 
        nn.ReLU()
    )

    self.layer_one = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(128), 
        nn.ReLU()
    )

    self.res_block1 = ResBlock(128, 128)

    self.layer_two = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256), 
        nn.ReLU()
    )

    self.layer_three = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), 
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(512), 
        nn.ReLU()
    )

    self.res_block2 = ResBlock(512, 512)

    self.max_pool = nn.MaxPool2d(4,4)
    self.fc = nn.Linear(512, num_classes, bias=False)


  def forward(self, x):
    x = self.prep_layer(x)

    x = self.layer_one(x)
    R1 = self.res_block1(x)
    x = x + R1

    x = self.layer_two(x)

    x = self.layer_three(x)
    R2 = self.res_block2(x)
    x = x + R2

    x = self.max_pool(x)

    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return(x)
  


def CustomResNet():
    return ResNet()
