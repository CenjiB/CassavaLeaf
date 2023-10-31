import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # identifies edges within an image
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3,3), stride=1, padding=1) # increase out_channel if accuracy is poor
        self.act = nn.ReLU() # converts all negative array values to zero
        self.drop1 = nn.Dropout(0.3) # prevents overfitting by randomly ignoring neurons during training

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3,3), stride=1, padding=1)

        self.flatten = nn.Flatten() # converts image array to one dimension
        self.fc = nn.Linear(400 * 300 * 3, 5) # image dimensions * num of layers for rgb, num of cassava leave diseases (num of possible predictions)
        self.sigmoid = nn.Sigmoid() # returns a number between one and zero for each of the 5 diseases

    def forward(self, x): # ADD LAYERS HERE !!!!!!!!!
        x = self.act(self.conv1(x))
        x = self.drop1(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
