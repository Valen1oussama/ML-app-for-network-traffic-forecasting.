import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
#load model
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
model = AirModel()
PATH='mnist_ffn.pth'
model.load_state_dict(torch.load(PATH))
model.eval()    

#get data:
def transform_data(file,lookback):
    raw_df = pd.read_csv(file)
    raw_df.head()
    df = pd.DataFrame(pd.to_datetime(raw_df.StartTime))
# we can find 'AvgRate' is of two scales: 'Mbps' and 'Gbps'
    raw_df.AvgRate.str[-4:].unique()
# Unify AvgRate value
    df['AvgRate'] = raw_df.AvgRate.apply(lambda x:float(x[:-4]) if x.endswith("Mbps") else float(x[:-4])*1000)
    df["total"] = raw_df["total"]
    dataset = df[["AvgRate"]].values.astype('float32')
    
    
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)

    return torch.tensor(np.array(X)), torch.tensor(np.array(y))
#predict
def get_prediction(x):
    predictions=model(x)
    return predictions


def rmse(x,y):  
 loss_fn = nn.MSELoss()
 with torch.no_grad():
  return  np.sqrt(loss_fn(x,y))


