from model.network import *
from utils.utils import *
from augment.augment import *
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

trainloader = get_loader(transform=None)
mean, std = get_stats(trainloader)
denorm = UnNormalize(mean, std)


train_transform = get_train_transform(mean, std)
test_transform = get_test_transform(mean, std)

trainloader = get_loader(transform=train_transform)
testloader = get_loader(transform=test_transform, train=False)

device = get_device()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

model = CustomResNet().to(device)
print(get_summary(model, device))


model = CustomResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

max_lr = LR_Finder(model, criterion, optimizer, trainloader)
print(f"Max LR @ {max_lr}")


scheduler = OneCycleLR(optimizer, max_lr=max_lr, 
                       epochs=epochs, steps_per_epoch=len(trainloader), 
                       pct_start=5/epochs,
                       div_factor=100,
                       three_phase=False,
                       )

def main():
    train_model(model, device, trainloader, testloader, criterion, scheduler, optimizer, epochs, max_lr)


if __name__ == "__main__":
    main()
