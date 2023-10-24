# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:40:11 2022

@author: 80594
"""
import os
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
Transform = transforms.Compose([transforms.Resize((48,48)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                                ])
class GTSRB_Test_Loader(Dataset):
    '''
    TEST_PATH:
        should be the path you reserve your test image. For example 'GTSRB/Final_Test/Images/'
    TEST_GT_PATH
        
    '''
    def __init__(self, TEST_PATH = None, TEST_GT_PATH = 'evaluation/GTSRB_Test_GT.csv'):
        self.df = pd.read_csv(TEST_GT_PATH,sep=';')
        self.TEST_PATH = TEST_PATH
        self.Transform = Transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = os.path.join(self.TEST_PATH, row['Filename'])
        img = Image.open(filename)

        img = self.Transform(img)
        
        label = int(row['ClassId'])
        return img, label, filename