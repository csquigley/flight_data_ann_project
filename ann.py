import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self,drop_p=0.5):
        super.__init__()
        #embedding sizes depends on data
        #you could set this up so that it is easy to take any kind of data in
        #and then spit out a single continuous value prediction
        embedding_sizes = [(2,4),(24,12),(24,12)]
        self.num_cat_embed = sum([nf for ne,nf in embedding_sizes])
        self.embeddings = nn.ModuleList([nn.Embedding(nf,ne) for nf, ne in embedding_sizes])
        #batch norm requires a number of input features as a parameter  
        self.batn1 = nn.BatchNorm1d(self.num_cat_embed)
        self.batn2 = nn.BatchNorm1d((self.num_cat_embed+num_continuous))
        self.drop1 = nn.Dropout(drop_p)
        self.input_layer = nn.Linear((self.num_cat_embed+num_continuous),out_feat)
        self.h_layer1 = nn.Linear(in_feat,out_feat)
        self.h_layer2 = nn.Linear(in_feat,out_feat)
        self.out = nn.Linear(in_feat,1)
        
    def forward(self,x_con,x_cat):
        #for your catagorical data 
        #first encode the data?
        ecat = []
        for i, e in enumerate(self.embeddings):
            ecat.append(e(x_cat[:,i]))
        #after encoding the data use torch.cat(data,1) to flatten the tensors
        ecat = torch.cat(ecat,1)
        #then you can use batch normalization
        ecat = self.batn1(ecat)
        #then you can use dropout
        ecat = drop1(ecat)
        #then you use torch.cat([cont_data,cat_data],1) to combine your categorical data and your continuous data
        x = torch.cat([ecat,x_con],1)
        #then you use batch normalization again on all the data
        x = nn.ReLU(self.h_layer1(x))
        x = nn.ReLu(self.h_layer2(x))
        x = self.out(x)
        
        #then you pass the data through the layers defined above, as well as an activation function
        
        #finally you get an output which you return as x
        
        return x