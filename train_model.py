import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Step 1: Download stock data (e.g., AAPL)
print("Downloading stock data...")
data = yf.download('AAPL', start='2015-01-01', end='2023-12-31')
close_prices = data['Close'].values.reshape(-1, 1)

# Step 2: Normalize prices
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(close_prices)

# Step 3: Prepare data for LSTM
X, y = [], []
seq_len = 60  # use last 60 days to predict next day
for i in range(seq_len, len(scaled_prices)):
    X.append(scaled_prices[i - seq_len:i])
    y.append(scaled_prices[i])
X, y = np.array(X), np.array(y)

# Step 4: Build model
print("Building model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Step 5: Compile & train
model.compile(optimizer='adam', loss='mean_squared_error')
print("Training model (this may take a minute)...")
model.fit(X, y, epochs=10, batch_size=32)

# Step 6: Save model
model.save("Stock_Predictor.keras")
print("✅ Model saved as 'Stock_Predictor.keras'")
