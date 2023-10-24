# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:19:21 2022

@author: 80594
"""

import torch
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from evaluation import evaluate

if __name__ == '__main__':
    torch.manual_seed(118)
    testloader = DataLoader(GTSRB_Test_Loader(), 
                    batch_size=50, 
                    shuffle=True, num_workers=8)
    model = None
    # import your trained model 
    testing_accuracy = evaluate(model, testloader)
    print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))