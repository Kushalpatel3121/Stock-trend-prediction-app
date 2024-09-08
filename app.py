import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.api.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
ticker = yf.Ticker(user_input)
if ticker:
    df = ticker.history('5y')
else:
    st.error('Could not find the ticker')

st.subheader('Summary of past 5 years of data')
st.write(df.describe())

st.subheader('Closing Price Vs Time Chart')
figure = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(figure)

# Visualization - with MA100
st.subheader('Closing Price Vs Time Chart with MA100')
ma100 = df.Close.rolling(100).mean()
figure = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(figure)

# Visualization - with MA200
st.subheader('Closing Price Vs Time Chart with MA100 and MA200')
ma200 = df.Close.rolling(200).mean()
figure = plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.plot(df.Close)
plt.legend()
st.pyplot(figure)

# Splitting Data for training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.8)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.8):len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Splitting data into x_train and y_train
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
model = load_model('lstm_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing])
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Predicted Graph
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)