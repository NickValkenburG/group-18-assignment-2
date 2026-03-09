import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torch import manual_seed, nn, no_grad, optim
device = 'cuda'  
torch.set_default_device(device)

torch.manual_seed(10101010) # because we randomly sort the data inside the data loader




width = 10
epochs = 100
print_every = 1
lr = 5e-3
batch_size = 20


def train_model(
    train_data,
    test_data,
    test_labels:bool,
    model,
    loss_fn,
    epochs:int,
    lr:float,
    batch_size:int,
    print_every:int,
    accuracy_fn=None,
):
    loss_dict = {"train": [], "test": [], "test_acc": []}

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_length = train_data[0].shape[0]
    
    # Print header.
    print(f"Epoch    Train loss      Test loss       Test accuracy")
    for epoch in range(epochs):
        epoch_loss_sum = 0
        models = []
        nr_batches = int(data_length/batch_size)
        for i in range(nr_batches): # only do full batches
            x_batch = train_data[0][i:i+batch_size,:]
            y_batch = train_data[1][i:i+batch_size]
            L_batch = train_data[2][i:i+batch_size]
            # Reset optimizer gradients.
            optimizer.zero_grad()
     
            # Predict the output
            y_pred = model(x_batch)

            # Compute the loss
            loss = loss_fn(y_pred, y_batch)
            epoch_loss_sum += loss.item()

            # Compute gradients according to newly computed loss.
            loss.backward()

            # Update the model parameters.
            optimizer.step()

        loss_dict["train"].append(epoch_loss_sum / nr_batches)

        with no_grad():
            test_pred = model(test_data[0])
            if test_labels:
                test_loss = loss_fn(test_pred, test_data[2]) #if we are training for the fence encloseing we do this
                test_accuracy = accuracy_fn(test_pred, test_data[2])
                loss_dict["test_acc"].append(test_accuracy.item())
            else:
                test_loss = loss_fn(test_pred, test_data[1]) #if we are training for the maximum area
                loss_dict["test_acc"].append(0.0)
            loss_dict["test"].append(test_loss.item())
                

        if (epoch + 1) % print_every == 0:
            
            print(
                f"{epoch+1: <7}  {loss_dict['train'][-1]: <14.6e}  {loss_dict['test'][-1]: <13.6e}  {loss_dict['test_acc'][-1]: .6}"
            )
            models.append(model)
            

    return models, loss_dict



def make_model_MAEBAPF(width = 8):  #Maximum area enclosed by a polygonal fence
    return nn.Sequential(
        nn.Linear(8, width, bias=True),
        nn.ReLU(),
        nn.Linear(width, width, bias=True),
        nn.ReLU(),
        nn.Linear(width, 1, bias=True),
    ) # with this model we are going for a dimension less one


def make_model_IPFCE():  #Is polygonal fence Center-Enclosing
    return nn.Sequential(
        nn.Linear(8, width, bias=True),
        nn.ReLU(),
        nn.Linear(width, width, bias=True),
        nn.ReLU(),
        nn.Linear(width, 1, bias=True),
        nn.Sigmoid(),
    )# with this model we are going for a dimension less one



class TensorData(Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.model.to('cuda') 
        self.input = input_tensor
        self.labels = label_tensor

    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, index):
        return self.input[index], self.labels[index]

def load_data_pd():
    fence5 = "kaggle_train_5_fences.csv"
    fence7 = "kaggle_train_7_fences.csv"
    fence9 = "kaggle_train_9_fences.csv"
    return {"fence5": pd.read_csv(fence5), "fence7": pd.read_csv(fence7), "fence9": pd.read_csv(fence9)}


#Load Data numpy

data_pd = load_data_pd()
data_pd_5 = data_pd["fence5"].copy()
data_pd_7 = data_pd["fence7"].copy()
data_pd_9 = data_pd["fence9"].copy()

#Padding fence5 and fence7 with zeros

data_pd_5.insert(6, '8', 0)
data_pd_5.insert(6, '7', 0)
data_pd_5.insert(6, '6', 0)
data_pd_5.insert(6, '5', 0)

data_pd_7.insert(8, '8', 0)
data_pd_7.insert(8, '7', 0)

#Isolating only the inputs

data_pd_5_train = data_pd_5.loc[:, ['0', '1', '2', '3', '4', '5', '6', '7', '8']]
data_pd_5_train.columns = pd.to_numeric(data_pd_5_train.columns)

data_pd_7_train = data_pd_7.loc[:, ['0', '1', '2', '3', '4', '5', '6', '7', '8']]
data_pd_7_train.columns = pd.to_numeric(data_pd_7_train.columns)

data_pd_9_train = data_pd_9.loc[:, ['0', '1', '2', '3', '4', '5', '6', '7', '8']]
data_pd_9_train.columns = pd.to_numeric(data_pd_9_train.columns)

#Sort function (smallest to biggest) for the fence lengths

def sort_fences(data):
    data_trans = data.copy()
    data_trans = data_trans.T
    for col in data_trans:
        data_trans[col] = data_trans[col].sort_values(ignore_index=True)
    return data_trans.T

#Sort data
data_pd_5_train_sorted = sort_fences(data_pd_5_train)

data_pd_7_train_sorted = sort_fences(data_pd_7_train)

data_pd_9_train_sorted = sort_fences(data_pd_9_train)

#Full dataframes with sorted values
data_pd_5_sorted = data_pd_5.copy()
data_pd_5_sorted[['0', '1', '2', '3', '4', '5', '6', '7', '8']] = data_pd_5_train_sorted[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
#data_pd_5_sorted.rename(columns={5: '5', 6: '6', 7: '7', 8: '8'})

data_pd_7_sorted = data_pd_7.copy()
data_pd_7_sorted[['0', '1', '2', '3', '4', '5', '6', '7', '8']] = data_pd_7_train_sorted[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
#data_pd_7_sorted.rename(columns={7: '7', 8: '8'})

data_pd_9_sorted = data_pd_9.copy()
data_pd_9_sorted[['0', '1', '2', '3', '4', '5', '6', '7', '8']] = data_pd_9_train_sorted[[0, 1, 2, 3, 4, 5, 6, 7, 8]]

data_pd_combined = pd.concat([data_pd_5_sorted, data_pd_7_sorted, data_pd_9_sorted])


#Homogenize the data

#Fence 5
f0 = data_pd_combined['0']
f1 = data_pd_combined['1']
f2 = data_pd_combined['2']
f3 = data_pd_combined['3']
f4 = data_pd_combined['4']
f5 = data_pd_combined['5']
f6 = data_pd_combined['6']
f7 = data_pd_combined['7']
f8 = data_pd_combined['8']

#Homogeneous inputs (multiply biggest (f8) by everything else; multiply guaranteed non-zeros (f8-f4 incl.))
x0 = f0*f8 #? should work maybe
x1 = f1*f8
x2 = f2*f8
x3 = f3*f8
x4 = f4*f8
x5 = f5*f8
x6 = f6*f8
x7 = f7*f8
x8 = f8*f8

#Homo data
data_nh = data_pd_combined.copy()
data_nh['0'] = x0
data_nh['1'] = x1
data_nh['2'] = x2
data_nh['3'] = x3
data_nh['4'] = x4
data_nh['5'] = x5
data_nh['6'] = x6
data_nh['7'] = x7
data_nh['8'] = x8


data_nd = data_pd_combined.copy()
data_nd['0'] = x0
data_nd['1'] = x1
data_nd['2'] = x2
data_nd['3'] = x3
data_nd['4'] = x4
data_nd['5'] = x5
data_nd['6'] = x6
data_nd['7'] = x7
data_nd['8'] = x8


#Homo data scaled -> only inputs
data_nh_scaled = data_nh.loc[:, ['0', '1', '2', '3', '4', '5', '6', '7', '8']]
data_nh_scaled = data_nh_scaled.div(data_nh['area'], axis=0)


#Dimensionless data -> inputs + area (divided by x8, so new inputs are x0 to x7)
data_nd = data_nh.loc[:, ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'area']]
data_nd = data_nd.div(data_nh['8'], axis=0)
data_nd['CE'] = data_nh['CE']





data_nd = data_nd.reset_index(drop=True)
data_nd_train = data_nd.sample(frac = 0.8 , random_state=1111,ignore_index=True)
data_nd_test = data_nd.drop(data_nd_train.index).reset_index(drop=True)




data_nh = data_nh.reset_index(drop=True)
data_nh_train = data_nh.sample(frac = 0.8 , random_state=1111,ignore_index=True)
data_nh_test = data_nh.drop(data_nd_train.index).reset_index(drop=True)

data_nh_scaled = data_nh_scaled.reset_index(drop=True)
data_nh_scaled_train = data_nh_scaled.sample(frac = 0.8 , random_state=1111,ignore_index=True)
data_nh_scaled_test = data_nh_scaled.drop(data_nd_train.index).reset_index(drop=True)

#turning the training data to torch tensors

data_nd_train_x = data_nd_train[['0','1','2','3','4','5','6','7']]
data_nd_train_y = data_nd_train['area']
data_nd_train_L = data_nd_train['CE']

data_nd_test_x = data_nd_test[['0','1','2','3','4','5','6','7']]
data_nd_test_y = data_nd_test['area']
data_nd_test_L = data_nd_test['CE']

tensor_nd_train_x = torch.from_numpy(data_nd_train_x.to_numpy().copy()).type(torch.float32).to('cuda') #x
tensor_nd_train_y = torch.from_numpy(data_nd_train_y.to_numpy().copy()).type(torch.float32).to('cuda') #y
tensor_nd_train_L = torch.from_numpy(data_nd_train_L.to_numpy().copy()).type(torch.float32).to('cuda') #label
tensor_nd_train = (tensor_nd_train_x,tensor_nd_train_y,tensor_nd_train_L)

tensor_nd_test_x = torch.from_numpy(data_nd_test_x.to_numpy().copy()).type(torch.float32).to('cuda') 
tensor_nd_test_y = torch.from_numpy(data_nd_test_y.to_numpy().copy()).type(torch.float32).to('cuda')
tensor_nd_test_L = torch.from_numpy(data_nd_test_L.to_numpy().copy()).type(torch.float32).to('cuda')
tensor_nd_test = (tensor_nd_test_x,tensor_nd_test_y,tensor_nd_test_L)

#tensor_nh_train = torch.from_numpy(data_nh_train.to_numpy() )
#tensor_nh_test = torch.from_numpy(data_nh_test.to_numpy() )

#tensor_nh_scaled_train = torch.from_numpy(data_nh_scaled_train.to_numpy())
#tensor_nh_scaled_test = torch.from_numpy(data_nh_scaled_test.to_numpy())








model = make_model_MAEBAPF(width)
labels = False
loss_fn = nn.MSELoss() # same loss function as in assignment 1
loss_fn = nn.L1Loss() # different loss function dont know which one to use
models_MAEBAPF, loss_dict_MAEBAPF = train_model(tensor_nd_train,tensor_nd_test,labels,model,loss_fn,epochs,lr,batch_size,print_every)
