# CIFAR10 Image Classification with PyTorch

This project implements an image classifier trained on the CIFAR10 dataset using PyTorch. The project aims to showcase the use of ResNet architecture, data augmentation, custom dataset classes, and learning rate schedulers.

## Project Structure

The project is structured as follows:

1. Data loading and preprocessing
2. Dataset statistics calculation
3. Data Augmentation
3. Model creation
4. Training and evaluation

### Data Loading and Preprocessing

The data for this project is the CIFAR10 dataset, which is loaded using PyTorch's built-in datasets. To ensure that our model generalizes well, we apply several data augmentations to our training set including normalization, padding, random cropping, and horizontal flipping.

### Dataset Statistics Calculation

Before we start training our model, we calculate per-channel mean and standard deviation for our dataset. These statistics are used to normalize our data, which helps make our training process more stable.

```
Dataset Mean - [0.49139968 0.48215841 0.44653091]
Dataset Std - [0.24703223 0.24348513 0.26158784] 
```

### Dataset Augmentation
```python
def get_transforms(means, stds):
  train_transforms = A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          A.RandomCrop(height=32, width=32, pad=4, always_apply=True),
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
```
![image](https://github.com/Delve-ERAV1/S10/assets/11761529/a0098b5b-e9d4-448b-a6c1-4b24ea9bdd98)


### Model Creation

The model we use for this project is a Custom ResNet, a type of convolutional neural network known for its high performance on image classification tasks.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
         ResBlock-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         294,912
        MaxPool2d-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
             ReLU-18            [-1, 256, 8, 8]               0
           Conv2d-19            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
             ReLU-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
         ResBlock-29            [-1, 512, 4, 4]               0
        MaxPool2d-30            [-1, 512, 1, 1]               0
           Linear-31                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.75
Params size (MB): 25.07
Estimated Total Size (MB): 31.84
----------------------------------------------------------------
```

#### ResNet Architecture and Residual Blocks

The defining feature of the ResNet architecture is its use of residual blocks and skip connections. Each residual block consists of a series of convolutional layers followed by a skip connection that adds the input of the block to its output. These connections allow the model to learn identity functions, making it easier for the network to learn complex patterns. This characteristic is particularly beneficial in deeper networks, as it helps to alleviate the problem of vanishing gradients.

### Training and Evaluation

To train our model, we use the Adam optimizer with a OneCycle learning rate scheduler. 

```
EPOCH: 18
Loss=0.0823458880186081 Batch_id=195 Accuracy=98.17: 100% 196/196 [00:19<00:00, 10.11it/s]

Test set: Average loss: 0.0011, Accuracy: 9215/10000 (92.15%)

EPOCH: 19
Loss=0.043348394334316254 Batch_id=195 Accuracy=98.71: 100% 196/196 [00:19<00:00, 10.18it/s]

Test set: Average loss: 0.0010, Accuracy: 9241/10000 (92.41%)

EPOCH: 20
Loss=0.019386257976293564 Batch_id=195 Accuracy=99.01: 100% 196/196 [00:19<00:00, 10.07it/s]

Test set: Average loss: 0.0010, Accuracy: 9290/10000 (92.90%)

EPOCH: 21
Loss=0.07089636474847794 Batch_id=195 Accuracy=99.13: 100% 196/196 [00:19<00:00, 10.17it/s]

Test set: Average loss: 0.0009, Accuracy: 9284/10000 (92.84%)

EPOCH: 22
Loss=0.021231239661574364 Batch_id=195 Accuracy=99.29: 100% 196/196 [00:19<00:00, 10.07it/s]

Test set: Average loss: 0.0009, Accuracy: 9312/10000 (93.12%)

EPOCH: 23
Loss=0.03242049738764763 Batch_id=195 Accuracy=99.33: 100% 196/196 [00:19<00:00, 10.07it/s]

Test set: Average loss: 0.0010, Accuracy: 9305/10000 (93.05%)

```

#### OneCycle Learning Rate Scheduler

The OneCycle learning rate scheduler varies the learning rate between a minimum and maximum value according to a certain policy. This dynamic learning rate can help improve the performance of our model. We train our model for a total of 24 epochs.

### Learning Rate Finder

```python
def LR_Finder(model, criterion, optimizer, trainloader):

  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, end_lr=10, num_iter=200, step_mode='exp')
  max_lr = lr_finder.plot(suggest_lr=True, skip_start=0, skip_end=0)
  lr_finder.reset()
  
  return(max_lr[1])
```

![image](https://github.com/Delve-ERAV1/S10/assets/11761529/fc84e96f-339f-4f1d-99b7-0fe726e8de02)


## Dependencies

This project requires the following dependencies:

- torch
- torchvision
- numpy
- albumentations
- matplotlib
- torchsummary

## Usage

To run this project, you can clone the repository and run the main script:

```bash
git clone https://github.com/Delve-ERAV1/S10.git
cd repo
python main.py
```

## Results

The model achieves an accuracy of approximately 93% on the CIFAR10 test set. Please note that due to random factors in the training process, results may vary slightly each time the script is run.

## References

Deep Residual Learning for Image Recognition Kaiming He et al
Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates Leslie N. Smith


