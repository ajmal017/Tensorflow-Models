import tensorflow as tf
from keras.layers import CuDNNLSTM, Dropout,BatchNormalization
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import datetime
import os
from tensorflow import keras
from keras.models import Sequential
from  keras.layers import LSTM, Dense, Flatten


def create_data_sets(tickers, test_portion=0.7):
    """
    ticker = python list of ticker from quandl
    test_portion = float decimal value to indicate what percentage is for training
    
    function takes the list input of ticker that will create
    each column of a pandas dataframe consisting of the adjusted open price.
    Then we calculate the percent change for each day.
    The security that has the greatest percent change for each day is recorded. 
    These values are then one hot encoded and returned as ndarray
    
    Returns four arrays x_train,y_train,x_test,y_test
    """
    df = pd.DataFrame()
    for ticker in tickers:#cycle through tickers
        call_ticker = "WIKI/"+ticker
        data = quandl.get(call_ticker)#use quandl api 
        series = data['Adj. Open']
        df[ticker] = series#add to dataframe
    time_max = df.first_valid_index()
    
    for column in df:#check each column most recent time
        time = df[column].first_valid_index()
        if time_max < time:
            time_max = time#find max time for entire dataframe
    df = df.loc[time_max:]
    df_pc = df.pct_change().fillna(value=0)#create percent change DF
    ticker_of_max = df_pc.idxmax(axis=1)#get the max value in each col
    correct = []
    for tic in ticker_of_max:
        correct.append(tickers.index(tic))#make list of max values
    df['correct'] = correct
    df['correct'] = df['correct'].shift().fillna(value=0)
    num_train_days = int(df['correct'].count()*test_portion)
    values = df.values     
    train = values[:num_train_days,:]
    test = values[num_train_days:,:]
    #cut the last column off for the y values
    x_train, y_train = train[:,:-1], train[:,-1]
    x_test, y_test = test[:,:-1], test[:,-1]
    return(x_train, to_categorical(y_train), x_test, to_categorical(y_test))


def reshape_scale(numpy_array):
    """
    normalize the x data sets 
    reshape into 3D array required input to Keras
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numpy_array)
    return scaled_array.reshape((scaled_array.shape[0], 1, scaled_array.shape[1]))
    
def graph_lines(df):
    """
    Vislize function that will graph all columns in a dataframe
    """
    fig, ax = plt.subplots()#visualize the data prices vs time
    df.plot(ax=ax)
#fig.savefig('C:\Development\ResumeSite\mainpage\static\mainpage')

def graph_pct_change(df):
    fig, ax = plt.subplots()#compare volatility of prices over time
    df.pct_change().plot(ax=ax)
    #fig.savefig('C:\Development\ResumeSite\mainpage\static\mainpage')



tickers = ['AAPL','GOOGL','FB','AMZN','SCHW','ETFC','EBAY','FB','F','HAL']
x_train, y_train ,x_test ,y_test = create_data_sets(tickers=tickers)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_new= x_train[:1000,:]
y_new = y_train[:1000,:]
X_new= x_test[:400,:]
Y_new = y_test[:400,:]

x_newer = x_new.reshape(5,200,9)
y_newer = y_new.reshape(5,200,10)
X_newer = X_new.reshape(2,200,9)
Y_newer = Y_new.reshape(2,200,10)

samples = 10
time_steps = 200
feats = 9
out_size = 10
node_size = 64
epochs = 200


logs_dir = 'C://Development/logs/CuDNNLSTM/{}-epochs-{}-node_size'.format(epochs,node_size)+ datetime.datetime.now().strftime("%I%M%p%B%d%Y")
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

model = Sequential()
model.add(CuDNNLSTM(node_size, return_sequences=True, input_shape=(time_steps, feats)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(rate=0.4))

model.add(CuDNNLSTM(node_size, return_sequences=True, input_shape=(time_steps, feats)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(rate=0.4))

model.add(CuDNNLSTM(node_size, return_sequences=True, input_shape=(time_steps, feats)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(rate=0.5))

model.add(CuDNNLSTM(node_size, return_sequences=True, input_shape=(time_steps, feats)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(rate=0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='ADAM', metrics=['accuracy'])
model.fit(x_newer,  y_newer, epochs=epochs,validation_data=(X_newer,Y_newer),verbose=2,callbacks=[tensorboard_callback])     
