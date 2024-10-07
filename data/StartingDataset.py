import torchvision
import torch
import pandas as pd
from PIL import Image
import os
import PIL
import glob
import math

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 21,397 color images of cassava leaves, size of image changed to 400x300x3.
    """
    # Haleigh - "/Users/haleigh/haleigh/cassava-leaf-disease-classification/train.csv"
    # Daisy - "C:/Users/dwatt/Downloads/cassava-leaf-disease-classification/train.csv"
    # Benji - "/Users/benjicarrere/.kaggle/cassava-leaf-disease-classification/train.csv"

    def __init__(self, train = False): 
        self.dataframe = pd.read_csv("/Users/benjicarrere/.kaggle/cassava-leaf-disease-classification/train.csv")
        if train == True:
            self.offset = 0 # will add 0 to index line 34
            self.dataframe = self.dataframe.iloc[0:math.floor((0.8*len(self.dataframe)))]
        elif train == False:
            self.offset = math.floor(0.8*len(self.dataframe)) # sets offset to the index of a random validation image
            self.dataframe = self.dataframe.iloc[math.floor(0.8*len(self.dataframe)):]
        self.dataframe.reset_index(drop=True) # resets index to default (iloc function)
        # print(self.dataframe.head())
        self.image_id = self.dataframe["image_id"]
        self.label = self.dataframe["label"]


    def __getitem__(self, index):
        # for validation dataset, index of a random validation image from dataset that contained training and validation set 
        # is added to index of dataset that only contains validation set
        index += self.offset
        image_id = self.image_id[index] # index is a built in function
        # print("chk pt 1")
        label = self.label[index]
        # print("chk pt 2")
        # classification = (image_id, label)
        im = Image.open('/Users/benjicarrere/.kaggle/cassava-leaf-disease-classification/train_images/' + image_id)
        resize_img = im.resize((400, 300))
        convert_tensor = torchvision.transforms.ToTensor() # creates function that convert im tuple to tensor
        return(convert_tensor(resize_img), label)


    def __len__(self):
        return len(self.image_id)
    
