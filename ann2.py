#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


# In[29]:


#read in data
columns = ['blank','search_rank','dep_date','dep_city','arr_city', 'search_date','dep_hour','dep_minute','dep_ampm','carrier','price','remaining_t','arr_hour','arr_minute','arr_ampm','plus_days','duration_hours','duration_minutes','layovers','layover_cities','layover_duration','next_carrier','multiple_airlines']
data = pd.read_csv('flight_data_testfile.csv',names=columns,index_col=False)
data.drop('blank',axis=1,inplace=True)
data = data.dropna()
data = data[]
data.info()


# In[30]:


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
#remove white space error added to arrival city
def correct_arr_city(x):
    a = x['arr_city']
    a = a[1:]
    return a
#     a = a[:,1:]
#     a = pd.Series(a)
#     return a





# In[31]:


# apply the above functions to our data
data['dep_dt'] = data.apply(convert_to_dt,axis=1)
data['time_to_takeoff'] = data.apply(time_to_takeoff,axis=1)
data['total_duration'] = data.apply(total_duration,axis=1)
data['dep_weekday'] = data.apply(weekday,axis=1)
data['arr_city'] = data.apply(correct_arr_city,axis=1)


# In[32]:


print(data['time_to_takeoff'][1])


# In[33]:


# turn the layover cities into individual columns
lays = data.apply(layovers,axis=1)
lays = lays.to_numpy()
data['l1'] = lays[:,0]
data['l2'] = lays[:,1]
data['l3'] = lays[:,2]





# In[34]:


#categorical data

cat_columns = ['dep_city','arr_city','dep_hour','dep_ampm','carrier',
               'remaining_t','arr_hour','arr_ampm','plus_days',
              'layovers','l1','l2','l3','next_carrier','multiple_airlines','dep_weekday']
#continuous data
cont_columns = ['dep_minute','arr_minute','time_to_takeoff','total_duration','layover_duration']
#change the categorical columns into the catagory datatype
for cat in cat_columns:
    data[cat] = data[cat].astype('category')
#create a list of embedding sizes for the nn.Embedding layer
embedding_sizes = []
for cat in cat_columns:
    embedding_sizes.append((data[cat].nunique(),min(50,(data[cat].nunique()+1)//2)))


# In[ ]:





# In[35]:


#turn data into tensors
xconts = np.stack([data[col].values for col in cont_columns],1)
xconts = torch.tensor(xconts,dtype=torch.float)
xcats = np.stack([data[col].cat.codes.values for col in cat_columns],1)
xcats = torch.tensor(xcats,dtype=torch.int64)

#Y label data: predicting price
y_true = torch.tensor([data['price'].values],dtype=torch.float)


# In[36]:


y_true = y_true.reshape(-1,1)


# In[37]:


#create a dataloader object to make it easier to shuffle data after each epoch

class MyDataset(Dataset):
    def __init__(self,xconts,xcats,y_true):
        self.xconts = xconts
        self.xcats = xcats
        self.y_true=y_true
    def __len__(self):
        return len(self.y_true)
    def __getitem__(self,index):
        xco = self.xconts[index]
        xca = self.xcats[index]
        y = self.y_true[index]
        return xco,xca,y



# In[38]:


dataset = MyDataset(xconts,xcats,y_true)
x_loader = DataLoader(dataset,batch_size=200,shuffle=True)


# In[39]:


#Model Architecture
class Model(nn.Module):

    def __init__(self,x_con=5,h1=100,h2=200,h3=100,drop_p=0.5,emb_szs=embedding_sizes):
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

    def forward(self,xconts,xcats):
        ecat = []
        for i, e in enumerate(self.embeddings):
            ecat.append(e(xcats[:,i]))
        ecat = torch.cat(ecat,1)
        ecat = self.drop(ecat)
        x_con = self.bat_norm1(xconts)
        x = torch.cat([ecat,xconts],1)
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


# In[40]:


#create Model instance
model = Model()
#define the loss function
criterion = torch.nn.MSELoss()
#define the optimizer and connect it to the model
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#Epochs
epochs = 100


# In[41]:


#training
import time
losses = []

for i in range(epochs):
    stime = time.time()
    i += 1
    for xconts,xcats,y_true in x_loader:
    #generate a prediction
        y_pred = model.forward(xconts,xcats)
        #calculate a loss
        loss = criterion(y_pred,y_true)

        losses.append(loss)
        #back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ttime = time.time() - stime
    if i % 10 == 0:
        print(f"epoch {i} loss: {loss}  Dur: {ttime}")


# In[42]:


plt.plot(losses)


# In[43]:


torch.save(model.state_dict(),'flight_ann_test_3.pt')


# In[ ]:







# In[44]:


#need to create a test point


# In[46]:



#Use this to create a single test point for prediction
test_point = []
for c in cat_columns:
    ctz = data[c].cat.categories.to_list()
    print(ctz)
    inp = input(f'enter {c}: ')
    test_point.append(inp)
for c in cont_columns:
    print(f"{c}")
    inp = input(f'enter an integer: ')
    test_point.append(inp)





# In[ ]:





# In[47]:


def convert_to_input_tensor(l):
    tensor_cat_list = []
    tensor_cont_list = []
    #convert categorical variables. I considered enumerate here, but it doesn't work on a few.
    a = l[0]
    a = data['dep_city'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[1]
    a = data['arr_city'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[2]
    a = data['dep_hour'].cat.categories.to_list().index(float(a))
    tensor_cat_list.append(a)
    a = l[3]
    a = data['dep_ampm'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[4]
    a = data['carrier'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[5]
    a = data['remaining_t'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[6]
    a = data['arr_hour'].cat.categories.to_list().index(float(a))
    tensor_cat_list.append(a)
    a = l[7]
    a = data['arr_ampm'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[8]
    a = data['plus_days'].cat.categories.to_list().index(float(a))
    tensor_cat_list.append(a)
    a = l[9]
    a = data['layovers'].cat.categories.to_list().index(int(a))
    tensor_cat_list.append(a)
    a = l[10]
    a = data['l1'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[11]
    a = data['l2'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[12]
    a = data['l3'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[13]
    a = data['next_carrier'].cat.categories.to_list().index(a)
    tensor_cat_list.append(a)
    a = l[14]
    a = data['multiple_airlines'].cat.categories.to_list().index(bool(a))
    tensor_cat_list.append(a)
    a = l[15]
    a = data['dep_weekday'].cat.categories.to_list().index(int(a))
    tensor_cat_list.append(a)

    #for the continuous variables
    for i in range(16,21):
        a = float(l[i])
        tensor_cont_list.append(a)
    tensor_cat_list = torch.tensor(tensor_cat_list,dtype=torch.int64)
    tensor_cont_list = torch.tensor(tensor_cont_list,dtype=torch.float)
    return tensor_cat_list, tensor_cont_list




# In[48]:


cat_var, cont_var = convert_to_input_tensor(test_point)
cat_var = cat_var.reshape(1,-1)
cont_var = cont_var.reshape(1,-1)


# In[49]:


print(cat_var)
print(cont_var)


# In[50]:


with torch.no_grad():
    model.eval()
    price_pred = model.forward(cont_var,cat_var)


# In[51]:


#predict the price on the test point
price_pred


# In[ ]:


# In[ ]:





# In[ ]:

# In[ ]:
