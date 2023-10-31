import torchvision
import torch
import pandas as pd
from PIL import Image
import os
import PIL
import glob

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        self.dataframe = pd.read_csv("/Users/haleigh/haleigh/cassava-leaf-disease-classification/train.csv")
        self.image_id = self.dataframe["image_id"]
        self.label = self.dataframe["label"]

    def __getitem__(self, index):
        image_id = self.image_id[index]
        label = self.label[index]
        classification = (image_id, label)
        im = Image.open('/Users/haleigh/haleigh/cassava-leaf-disease-classification/train_images/' + image_id)
        resize_img = im.resize((400, 300))
        convert_tensor = torchvision.transforms.ToTensor() # creates function that convert im tuple to tensor
        return(convert_tensor(resize_img), label)
        # resize_img.show()

        return classification

    def __len__(self):
        return len(self.image_id)
    
