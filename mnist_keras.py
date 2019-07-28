##################################################################################################
#Keras mnist model
##################################################################################################


from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()



import matplotlib.pyplot as plt
#plot the first image in the dataset
plt.imshow(X_train[0])
plt.show()



#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)



from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]





from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
import os
import datetime
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

epochs = [1]
convo_layers = [3]
dense_layers = [0]
 

for epoch in epochs:
    for convo in convo_layers:
        for dense in dense_layers:
            logs_dir = 'C://Development/logs/MNIST_keras/mnist-{}-epochs-{}-convo_layers-{}-max_pooling-{}-dropout'.format(epoch,convo,convo,convo)+ datetime.datetime.now().strftime("%I%M%p%B%d%Y")
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)
                tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)
            #create model
            model = Sequential()
            #add model layers
            model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
            for layer in range(convo):
                model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
                model.add(Dropout(0.25))
            model.add(Flatten())
            for layer in range(dense):
                model.add(Dense(32))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch,callbacks=[tensorboard_callback])
            


for i in range(10):
    img = X_train[i:i+1].reshape(28,28)
    plt.imshow(img,'gray')
    plt.show()

