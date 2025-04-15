import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import warnings

# Load trained model
model = load_model("Stock_Predictor.keras")

# Streamlit UI
st.header('📈 Stock Market Predictor')

# User input

stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG, MSFT , NFLX , NVDA , TSLA , META ,ORCL)', 'GOOG')
start = '2012-01-01'
end = date.today().strftime('%Y-%m-%d')

# Fetch data
data = yf.download(stock, start, end)

# Show raw data
st.subheader('🔍 Stock Data')
st.write(data)

# Splitting training and test
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
final_test_scaled = scaler.fit_transform(final_test_data)

# Moving Averages: MA50
st.subheader('📊 Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, label='Closing Price')
plt.plot(ma_50_days, label='MA50', color='red')
plt.legend()
st.pyplot(fig1)

# MA50 vs MA100
st.subheader('📊 Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, label='Closing Price')
plt.plot(ma_50_days, label='MA50', color='red')
plt.plot(ma_100_days, label='MA100', color='blue')
plt.legend()
st.pyplot(fig2)

# MA100 vs MA200
st.subheader('📊 Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, label='Closing Price')
plt.plot(ma_100_days, label='MA100', color='red')
plt.plot(ma_200_days, label='MA200', color='blue')
plt.legend()
st.pyplot(fig3)

# Predicting
x_test = []
y_test = []

for i in range(100, final_test_scaled.shape[0]):
    x_test.append(final_test_scaled[i-100:i])
    y_test.append(final_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
predicted = model.predict(x_test)

# Scale back
scale_factor = 1 / scaler.scale_[0]
predicted = predicted * scale_factor
y_test = y_test * scale_factor

# Prediction plot
st.subheader('📉 Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Original Price', color='green')
plt.plot(predicted, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig4)
