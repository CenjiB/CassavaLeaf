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

    def __init__(self, train = False): 
        self.dataframe = pd.read_csv("/Users/haleigh/haleigh/cassava-leaf-disease-classification/train.csv")
        # FIRST ORDER OF BUSINESS 11/15 REDO SPLITTING OF DATA
        if train == True:
            self.dataframe = self.dataframe.iloc[0:math.floor((0.8*len(self.dataframe)))]
        elif train == False:
            self.dataframe = self.dataframe.iloc[math.floor(0.8*len(self.dataframe)):]
        self.image_id = self.dataframe["image_id"]
        self.label = self.dataframe["label"]


    def __getitem__(self, index):
        image_id = self.image_id[index] # IS THIS ACCESSING ALL DATA AS OPPOSED TO JUST TRAINING OR JUST VAL
        label = self.label[index]
        classification = (image_id, label)
        im = Image.open('/Users/haleigh/haleigh/cassava-leaf-disease-classification/train_images/' + image_id)
        resize_img = im.resize((400, 300))
        convert_tensor = torchvision.transforms.ToTensor() # creates function that convert im tuple to tensor
        return(convert_tensor(resize_img), label)


    def __len__(self):
        return len(self.image_id)
    
