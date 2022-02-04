import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import re
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#read in data
columns = ['blank','search_rank','dep_date','dep_city','arr_city', 'search_date','dep_hour','dep_minute','dep_ampm','carrier','price','remaining_t','arr_hour','arr_minute','arr_ampm','plus_days','duration_hours','duration_minutes','layovers','layover_cities','layover_duration','next_carrier','multiple_airlines']
data = pd.read_csv('/Users/christopherquigley/Desktop/FLIGHT_ANN/flight_data_ann_project/flight_data_testfile.csv',names=columns,index_col=False)
data.drop('blank',axis=1,inplace=True)
data = data.dropna()
data.info()

#a few functions for feaature engineering
#convert raw data into datetime objects
def convert_to_dt(x):
    r = pd.to_datetime(x['dep_date']+" "+str(x['dep_hour']) + ":" + str(x['dep_minute']))
    return r

#calculated the time from searching for a flight, to the takeoff time, in days
def time_to_takeoff(x):
    r = (x['dep_dt']-pd.to_datetime(x['search_date']))
    r = r.days + (r.seconds/86400)
    return r

#total trip duration
def total_duration(x):
    r = x['duration_hours'] + x['duration_minutes']/60
    return r

#cities where layovers take place
def layovers(x):
    l = pd.Series(['NONE','NONE','NONE'])
    r = re.findall("[A-Z]+",x['layover_cities'])
    for i, e in enumerate(r):
        l[i] = e
        if l[i] == 'N':
            l[i] = 'NONE'
    return l
#takeoff day of the week
def weekday(x):
    return x['dep_dt'].weekday()

#apply the above functions to our data
data['dep_dt'] = data.apply(convert_to_dt,axis=1)
data['time_to_takeoff'] = data.apply(time_to_takeoff,axis=1)
data['total_duration'] = data.apply(total_duration,axis=1)
data['dep_weekday'] = data.apply(weekday,axis=1)

#turn the layover cities into individual columns
lays = data.apply(layovers,axis=1)
lays = lays.to_numpy()
data['l1'] = lays[:,0]
data['l2'] = lays[:,1]
data['l3'] = lays[:,2]

#categorical data

cat_columns = ['dep_city','arr_city','dep_hour','dep_ampm','carrier',
               'remaining_t','arr_hour','arr_ampm','plus_days',
              'layovers','l1','l2','l3','next_carrier','multiple_airlines','dep_weekday']
#continuous data
cont_columns = ['dep_minute','arr_minute','total_duration','layover_duration']
#change the categorical columns into the catagory datatype
for cat in cat_columns:
    data[cat] = data[cat].astype('category')
#create a list of embedding sizes for the nn.Embedding layer
embedding_sizes = []
for cat in cat_columns:
    embedding_sizes.append((data[cat].nunique(),min(50,(data[cat].nunique()+1)//2)))

#shuffle data
#NOTE: Should shuffle data after each epoch for more robust results
data = data.sample(frac=1)

#turn data into tensors
xcont = np.stack([data[col].values for col in cont_columns],1)
xcont = torch.tensor(xcont,dtype=torch.float)
xcats = np.stack([data[col].cat.codes.values for col in cat_columns],1)
xcats = torch.tensor(xcats,dtype=torch.int64)

#Y label data: predicting price
y_true = torch.tensor([data['price'].values],dtype=torch.float)
y_true = y_true.reshape(-1,1)
#create a dataset object which is needed to create a dataloader object
class MyDataset(Dataset):
    def __init__(self,xcont,xcat,y_true):
        self.xcont = xcont
        self.xcat = xcat
        self.y_true=y_true
    def __len__(self):
        return len(self.y_true)
    def __getitem__(self,index):
        xco = self.xcont[index]
        xca = self.xcat[index]
        y = self.y_true[index]
        return xco,xca,y

dataset = MyDataset(xcont,xcats,y_true)
#create dataloader
x_loader = DataLoader(dataset,batch_size=500,shuffle=True)
#Model Architecture
#Model Architecture
class Model(nn.Module):

    def __init__(self,x_con=4,h1=100,h2=200,h3=100,drop_p=0.5,emb_szs=embedding_sizes):
        super().__init__()
        self.emb_szs = emb_szs
        self.num_cat_embed = sum([nf for ne,nf in emb_szs])
        self.embeddings = nn.ModuleList([nn.Embedding(nf,ne) for nf, ne in embedding_sizes])
        self.drop = nn.Dropout(drop_p)
        self.bat_norm1 = nn.BatchNorm1d(x_con)
        self.input_layer = nn.Linear((self.num_cat_embed+x_con),h1)
        self.h_layer1 = nn.Linear(h1,h2)
        self.bat_norm2 = nn.BatchNorm1d(h2)
        self.h_layer2 = nn.Linear(h2,h3)
        self.bat_norm3 = nn.BatchNorm1d(h3)
        self.out = nn.Linear(h3,1)

    def forward(self,xcon,xcat):
        ecat = []

        for i, e in enumerate(self.embeddings):
            ecat.append(e(xcat[:,i]))
        ecat = torch.cat(ecat,1)
        ecat = self.drop(ecat)
        x_con = self.bat_norm1(xcon)
        x = torch.cat([ecat,x_con],1)
        x = F.relu(self.input_layer(x))
        x = self.drop(x)
        x = F.relu(self.h_layer1(x))
        x = self.drop(x)
        x = self.bat_norm2(x)
        x = F.relu(self.h_layer2(x))
        x = self.drop(x)
        x = self.bat_norm3(x)
        x = self.out(x)
        return x
#create Model instance
model = Model()
#define the loss function
criterion = torch.nn.MSELoss()
#define the optimizer and connect it to the model
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#Epochs: 5000+ works well for final model
epochs = 200
losses = []

for i in range(epochs):
    for xcont,xcat,y_true in x_loader:
    #generate a prediction
        y_pred = model.forward(xcont,xcat)
        #calculate a loss
        loss = criterion(y_pred,y_true)

        losses.append(loss)
        #back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f"loss: {loss}")

plt.plot(losses)

torch.save(model.state_dict(),'flight_ann_test_3.pt')
