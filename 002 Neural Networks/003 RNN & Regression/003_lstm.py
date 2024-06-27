#==========================================#
# Title:  Stock prices prediction with LSTM
# Author: Jaewoong Han
# Date:   2024-06-05
#==========================================#
import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Objective: Predict Stock market price
 - Input: Close prices -> input_dim == 1
 - Output: Close price -> features == 1 / output_dim == 1
"""
#Download data from yfinance library
def download_data(ticker, start, end):
    yf.pdr_override()
    df = yf.download(ticker, start=start, end=end)
    print("="*50)
    print("Dataset summarize")
    print(df.describe())
    print("="*50)
    return df

# Data preprocessing
def preprocess_data(df):
    close_prices = df['Close'].values.reshape(-1, 1)  # Use close price only
    scaler = MinMaxScaler(feature_range=(0, 1))  # scale data to range 0-1
    scaled_data = scaler.fit_transform(close_prices)

    return scaled_data, scaler

# Create dataset
def create_dataset(data, time_step):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # reshape to [samples, time steps, features]
    return x, y

# Get and preprocess data
df = download_data('SBUX', '2019-01-01', '2023-12-31')
scaled_data, scaler = preprocess_data(df)

# Split data into train and test sets
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Create train and test datasets
time_step = 60
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        out, _ = self.lstm(x, (h0, c0)) # LSTM forward pass
        out = self.fc(out[:, -1, :]) # get output from the last time step
        return out

# Initialize and set up model for training
model = LSTMModel(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(20):
    for i in range(len(x_train)):
        inputs = Variable(torch.from_numpy(x_train[i].astype(np.float32)).unsqueeze(0))
        labels = Variable(torch.from_numpy(np.array([y_train[i]]).astype(np.float32)).unsqueeze(1))

        optimizer.zero_grad() # clear previous gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels) # compute loss
        loss.backward()  # backpropagate the error
        optimizer.step()  # update model parameters

# Make predictions
model.eval()
x_test_tensor = Variable(torch.from_numpy(x_test.astype(np.float32)))
predicted_stock_price = model(x_test_tensor)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.detach().numpy())

# Calculate RMSE
rmse = np.sqrt(np.mean(np.square(predicted_stock_price - y_test.reshape(-1,1))))
print(rmse)

# Plot the results
predicted_prices = pd.DataFrame(data={'Predictions': predicted_stock_price.flatten()})
GT = df.iloc[len(df) - len(predicted_prices):]
GT['Predictions'] = predicted_prices['Predictions'].values

plt.figure()
plt.plot(GT[['Close', 'Predictions']])
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend(['Ground Truth', 'Predictions'], loc='lower right')
plt.show()
