import pandas as pd
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from Utils.timeseries import add_lags
from torch.optim import Adam
from Utils.pytorch import train_pytorch


df_train = pd.read_csv(r'C:\Users\bigme\OneDrive\Documents\Personal GitHub\Koopman\Data\India Daily Climate Data\DailyDelhiClimateTrain.csv')
df_train_ = add_lags(df_train.iloc[:,1:], lag=2)
columns = df_train_.columns
train = torch.tensor(df_train_.iloc[:, 1:].values)
model = nn.Sequential(
    nn.Linear(4, 4)
)
optimizer = Adam(model.parameters())
loss = nn.MSELoss()
train_loader = torch.utils.data.DataLoader(train, shuffle=False)
train_pytorch(model, optimizer, loss, train_loader,100,0)