import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
columns = ['blank','search_rank','dep_date','dep_city','arr_city', 'search_date','dep_hour','dep_minute','dep_ampm','carrier','price','remaining_t','arr_hour','arr_minute','arr_ampm','plus days','duration_hours','duration_minutes','layovers','layover_cities','layover_duration','next_carrier','multiple_airlines']
data = pd.read_csv('/Users/christopherquigley/Desktop/FLIGHT_ANN/flight_data_ann_project/flight_data.csv',names=columns,index_col=False)
data.drop('blank',axis=1,inplace=True)
data.info()
#take a .look at the data
def convert_to_dt(x):
    r = pd.to_datetime(x['dep_date']+" "+str(x['dep_hour']) + ":" + str(x['dep_minute']))
    return r

def time_to_takeoff(x):
    r = (x['arr_dt']-pd.to_datetime(x['search_date']))
    r = r.days + (r.seconds/86400)
    return r
def total_duration(x):
    r = x['duration_hours'] + x['duration_minutes']/60
    return r
def layovers(x):
    r1,r2,r3 = re.findall("\[[/S]",x)

data['arr_dt'] = data.apply(convert_to_dt,axis=1)
data['time_to_takeoff'] = data.apply(time_to_takeoff,axis=1)
data['total_duration'] = data.apply(total_duration,axis=1)
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
