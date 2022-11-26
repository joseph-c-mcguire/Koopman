import pandas as pd
import torch
import torch.nn as nn
from Utils.timeseries import add_lags
from torch.optim import Adam
from Utils.pytorch import train_pytorch
from torch.utils.data import TensorDataset, DataLoader

df_train = pd.read_csv(r'C:\Users\bigme\OneDrive\Documents\Personal GitHub\Koopman\Data\India Daily Climate Data\DailyDelhiClimateTrain.csv')
df_train_ = add_lags(df_train.iloc[:, 1:], lag=0)
columns = df_train_.columns
train = torch.tensor(df_train_.iloc[:-1, :].values).to(torch.double)
target = torch.tensor(df_train.iloc[1:, 1:].values).to(torch.double)
model = nn.Sequential(
    nn.Linear(4, 4, dtype=torch.float64)
)
optimizer = Adam(model.parameters())
loss = nn.MSELoss()
train_loader = TensorDataset(train, target)
train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
train_pytorch(model, optimizer, loss, train_loader, 5, 100)

$