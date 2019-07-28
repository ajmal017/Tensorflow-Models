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
    input:
    ticker = python list of ticker from quandl
    test_portion = indicate what percentage is for training, default = 0.7
    
    function takes the list input of ticker that will create
    each column of a pandas dataframe consisting of the adjusted open prices.
    Then we calculate the percent change for each day.
    The security that has the greatest percent change for each day is recorded. 
    These values are then one hot encoded and returned as ndarray
    
    Output: four arrays x_train,y_train,x_test,y_test
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


def LSTM_node_test():
    """ 
    Expirement that test different number of nodes in only one LSTM
    Loops through different node sizes 32 64 128 for 1 lstm layer
    """
    for node_size in (32,64,128,256):
        logs_dir = 'C://Development/logs/LSTM_node_size/{}_nodes_1_lstm'.format(node_size) + datetime.datetime.now().strftime("%I%M%p%B%d%Y")
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)
        model = Sequential()
        model.add(LSTM(node_size, input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dense(len(y_train[0])))
        model.compile(loss='categorical_crossentropy',optimizer='ADAM', metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=30,batch_size=10,validation_data=(x_test,y_test),verbose=2,callbacks=[tensorboard_callback])



def LSTM_layer_test():
    """ 
    Expirement that tests how many LSTM layers is optimal
    This will use the optimal node size from LSTM_node_test
    Sizes (1,2,3,4,5)
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
    
    node_size = 256 
    layers = [1,2,3,4,5]
    
    for layer in layers:
        logs_dir = 'C://Development/logs/LSTM_layer_test/{}node size-{}-LSTM_layers'.format(node_size,layer)+ datetime.datetime.now().strftime("%I%M%p%B%d%Y")
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)
        model = Sequential()
        if layer == 1:
            model.add(LSTM(node_size, input_shape=(x_train.shape[1],x_train.shape[2])))
            model.add(Dense(len(y_train[0])))
        else:
            for num in range(layer):
                print(num)
                if num == layer-1:
                    model.add(LSTM(node_size, input_shape=(x_train.shape[1],x_train.shape[2])))
                    model.add(Dense(len(y_train[0])))
                else:
                    model.add(LSTM(node_size,return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
            
        model.compile(loss='categorical_crossentropy',optimizer='ADAM', metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=150,batch_size=10,validation_data=(x_test,y_test),verbose=2,callbacks=[tensorboard_callback])     


tickers = ['AAPL','GOOGL','FB','AMZN','SCHW','ETFC','EBAY','FB','F','HAL']
x_train, y_train ,x_test ,y_test = create_data_sets(tickers=tickers)
x_train = reshape_scale(x_train)
x_test = reshape_scale(x_test)


