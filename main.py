from model.network import *
from utils.utils import *
from augment.augment import *
from dataset.dataset import *
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pprint import pprint
from torch_lr_finder import LRFinder

config = process_config("utils/config.yml")
pprint(config)

classes = config["data_loader"]["classes"]
batch_size = config["data_loader"]['args']["batch_size"]
num_workers = config["data_loader"]['args']["num_workers"]
dropout = config["model_params"]["dropout"]
seed = config["model_params"]["seed"]
epochs = config["training_params"]["epochs"]


#####################################

SEED = 42
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
  torch.cuda.manual_seed(SEED)

# dataloader arguments
dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)


train = CIFAR10Dataset(transform=None)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
mu, std = get_stats(train_loader)
train_transforms, test_transforms = get_transforms(mu, std)

# train dataloader
train = CIFAR10Dataset(transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)


# test dataloader
test = CIFAR10Dataset(transform = test_transforms, train=False)
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


# get some random training images
images, labels = next(iter(train_loader))
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# show images
imshow(torchvision.utils.make_grid(images[:4]))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


###########################

device = get_device()


model = CustomResNet().to(device)
print(get_summary(model, device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

max_lr = LR_Finder(model, criterion, optimizer, train_loader)
print(f"Max LR @ {max_lr}")


scheduler = OneCycleLR(optimizer, max_lr=max_lr, 
                       epochs=epochs, steps_per_epoch=len(train_loader), 
                       pct_start=5/epochs,
                       div_factor=100,
                       three_phase=False,
                       )

def main():
  
    train_losses, train_acc = [], []
    test_losses, test_acc = [], []

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)

        train(model, device, train_loader, criterion, scheduler, optimizer, [train_losses, train_acc])
        test(model, device, test_loader, criterion, [test_losses, test_acc])


if __name__ == "__main__":
    main()
