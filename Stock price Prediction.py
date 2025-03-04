## 6. Stock Price Prediction

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('stock_prices.csv')
data = data[['Close']]
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Prepare dataset
X, y = [], []
for i in range(10, len(data)):
    X.append(data[i-10:i, 0])
    y.append(data[i, 0])
X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16)
