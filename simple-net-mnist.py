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
            
######################################################
######################################################
def accuracy(predictions,labels):
    return np.sum(np.argmax(predictions,axis=1)==np.argmax(labels,axis=1))*100/labels.shape[0]

batch_size = 100
layer_names = ['layer1','layer2','layer3','out']
layer_sizes = [784,200,200,200,10]

tf.reset_default_graph()

image_holder = tf.placeholder(tf.float32,shape=[batch_size,784],name='image_holder')
label_holder = tf.placeholder(tf.float32,shape=[batch_size,10],name='train_labels')

for index,layer in enumerate(layer_names):
    with tf.variable_scope(layer):
        w=tf.get_variable('weights',shape=[layer_sizes[index],layer_sizes[index+1]]
                         ,initializer=tf.truncated_normal_initializer(stddev=0.05))
        b=tf.get_variable('bias',shape=[layer_sizes[index+1]],
                         initializer=tf.random_uniform_initializer(-.1,0.1))
    

h = image_holder
for layer in layer_names:
    with tf.variable_scope(layer,reuse=True):
        w , b = tf.get_variable('weights'),tf.get_variable('bias')
        if layer !='out':
            h = tf.nn.relu(tf.matmul(h,w)+b,name=layer+'_output')
        else:
            h = tf.nn.xw_plus_b(h,w,b,name=layer+'_output')
            
predictions = tf.nn.softmax(h, name='predictions')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_holder, logits=h),name='loss')

learning_rate = tf.placeholder(tf.float32,shape=None,name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
gradient_vars = optimizer.compute_gradients(loss)
loss_minimized = optimizer.minimize(loss)

with tf.name_scope('performance'):
    loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    loss_summary = tf.summary.scalar('loss',loss_ph)
    accuracy_ph = tf.placeholder(tf.float32,shape=None,name='accuracy_summary')
    accuracy_summary = tf.summary.scalar('accuracy',accuracy_ph)
    
for gradient,var in gradient_vars:
    with tf.name_scope('gradients'):
        last_grad_norm = tf.sqrt(tf.reduce_mean(gradient**2))
        gradient_norm_summary = tf.summary.scalar('gradient_norm',last_grad_norm)
        break
performance_summaries = tf.summary.merge([
    loss_summary,accuracy_summary])

import os

image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 10000
n_epochs = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth =True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

session = tf.InteractiveSession(config=config)
sum_dir = 'C:\Development\logs\MNIST_runs'

if not os.path.exists(sum_dir):
    os.mkdir(sum_dir)

summary_writer = tf.summary.FileWriter(sum_dir,session.graph)

tf.global_variables_initializer().run()

accuracy_per_epoch = []
mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)

#train the model
for epoch in range(n_epochs):
    for i in range(n_train//batch_size):
        loss_per_epoch = []
        batch = mnist_data.train.next_batch(batch_size)

        l ,_= session.run([loss,loss_minimized],feed_dict={
            image_holder:batch[0].reshape(batch_size,image_size*image_size),
            label_holder:batch[1],
            learning_rate:.0001})
        loss_per_epoch.append(l)
    print('average loss in epoch {}: {}'.format(epoch,np.mean(loss_per_epoch)))
    avg_loss = np.mean(loss_per_epoch)

#calculate validation
validation_accuracy_per_epoch = []
for i in range(n_valid//batch_size):
    valid_images, valid_labels = mnist_data.validation.next_batch(batch_size)
    valid_batch_predictions = session.run(predictions,
                                    feed_dict={image_holder:valid_images.reshape(batch_size,image_size*image_size)})
validation_accuracy_per_epoch.append(accuracy(valid_batch_predictions,valid_labels))

mean_v_acc = np.mean(validation_accuracy_per_epoch)
print('average validation accuracy in epoch {}: {}'.format(epoch,np.mean(validation_accuracy_per_epoch)))
    

