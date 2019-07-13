import tensorflow as tf
import tensorflow_datasets as tf


import tensorflow as tf
from PIL import Image

import numpy as np

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print(train_data.shape , train_labels.shape, eval_data.shape, eval_labels.shape)


image_holder = tf.placeholder(tf.float32,[None,784])
label_holder = tf.placeholder(tf.float32,[None,10])

hid_nodes = 200
out_nodes = 10

w0 = tf.Variable(tf.random_normal([784, hid_nodes]))
w1 = tf.Variable(tf.random_normal([hid_nodes, hid_nodes]))
w2 = tf.Variable(tf.random_normal([hid_nodes, hid_nodes]))
w3 = tf.Variable(tf.random_normal([hid_nodes, out_nodes]))

b0 = tf.Variable(tf.random_normal([hid_nodes]))
b1 = tf.Variable(tf.random_normal([hid_nodes]))
b2 = tf.Variable(tf.random_normal([hid_nodes]))
b3 = tf.Variable(tf.random_normal([out_nodes]))

layer1 = tf.add(tf.matmul(image_holder, w0),b0)
layer1 = tf.nn.relu(layer1)
layer2 = tf.add(tf.matmul(layer1, w1),b1)
layer2 = tf.nn.relu(layer1)
layer3 = tf.add(tf.matmul(layer2, w2),b2)
layer3 = tf.nn.relu(layer3)
out_layer = tf.matmul(layer3,w3) +b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer,labels=label_holder))

learn_rate = .01
num_epochs = 1
batch_size = 100
num_batches = int(mnist.train.num_examples/batch_size)

optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto(device_count={'GPU':0})
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    sess.run(init)
    feed_dict = {image_holder:train_data,label_holder:train_labels}
    test_dict = {image_holder:eval_data,label_holder:eval_labels}
    for epoch in range(num_epochs):
        for batch in range(batch_size):
            sess.run(optimizer,feed_dict = feed_dict)
    prediction = tf.equal(tf.argmax(out_layer,1),tf.argmax(label_holder,1))
    sucess = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print("Success rate" ,sess.run(sucess,feed_dict=test_dict))
            
