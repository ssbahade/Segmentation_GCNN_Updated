
import matplotlib
import tensorflow as tf
import numpy as np
import scipy as sps
import math
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import pickle
from sklearn.metrics import confusion_matrix
from Create_patches import remove_unsized_patched
from Build_Graph import compute_weight_matrix


training_dir ='./Training_data'

# For Keras
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.image_ops_impl import ResizeMethod



# load images, divide into patches and return lost of patch_images and patch_labels
image_data, label_data = remove_unsized_patched(training_dir)
image_data = image_data.astype(float)
label_data = label_data.astype(float)

split = 9240                       # which is 30% of 30800 patches

# create Validation set
training_images = image_data[:-split]                   # training data
training_true_class = label_data[:-split]

validation_images = image_data[-split:]
validation_true_class = label_data[-split:]             # validation data

# The size of the image height and width
image_size = training_images.shape[1]
image_shape = (image_size, image_size)
image_size_flat = image_size * image_size

unpooled_graph_struct = compute_weight_matrix(image_size_flat, image_shape)
A = unpooled_graph_struct['weight-matrix']
D = np.array(np.sum(A, axis=0))
D_ = np.diagflat(D)

######################################################  define placeholders
features = tf.placeholder(tf.float32, shape=(32,32,3), name='features')
adjacency = tf.placeholder(tf.float32, shape=(1024,1024),name='adjacency')
degree = tf.placeholder(tf.float32, shape=(1024,1024),name='degree')
labels = tf.placeholder(tf.float32, shape=(32,32),name='label')

# print("shape: {}".format(true_class_reshape.get_shape()))

weights_1 = tf.Variable(tf.random_normal([3,16],stddev=1))
weights_2 = tf.Variable(tf.random_normal([16,3],stddev=1))

def layer(features, adjacency, degree, weights_1):
    with tf.name_scope('gcn_layer'):
        d_ = tf.pow(degree + tf.eye(1024),-0.5)
        y = tf.matmul(d_, tf.matmul(adjacency, d_))
        features_reshape = tf.reshape(features,shape=[-1,3])
        kernel = tf.matmul(features_reshape, weights_1)
        return tf.nn.relu(tf.matmul(y, kernel))

model_1 = layer(features, adjacency, degree, weights_1)                    # features = 1024*3, adjacency = 1024*1024, degree = 1024*1024, weights =

def layer(model_1, adjacency, degree, weights_2):
    with tf.name_scope('gcn_layer'):
        d_ = tf.pow(degree + tf.eye(1024),-0.5)
        y = tf.matmul(d_, tf.matmul(adjacency, d_))
        #features_reshape = tf.reshape(features,shape=[-1,3])
        kernel = tf.matmul(model_1, weights_2)
        return tf.nn.relu(tf.matmul(y, kernel))

model_2 = layer(model_1, adjacency, degree, weights_2)

print("model Shape: {}".format(model_1.get_shape()))

with tf.name_scope('loss'):
    true_class_reshape = tf.reshape(labels, shape=[-1])
    true_class_cast = tf.cast(true_class_reshape,tf.int32)
    true_one_hot = tf.one_hot(true_class_cast, 3)
    log = tf.argmax(tf.nn.softmax(model_2,axis=0),axis=1)
    lab = tf.argmax(true_one_hot,axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_2, labels=true_one_hot))
    print("logit shape: {} and label shape: {}".format(model_2.get_shape(),true_one_hot.get_shape()))
    train_op = tf.train.AdamOptimizer(0.001,0.9).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,len(training_images)):
        _,loss_ = sess.run([train_op, loss], feed_dict={features: training_images[i],adjacency: A, degree: D_,labels: training_true_class[i]})               # input feature: no.of nodes 32*32 and features on each node 3
        print("Loss: {}".format(loss_))                                                                                                                     # Adjacency matrix no.of nodes * no.of nodes 1024 * 1024
                                                                                                                                                            # Degree matrix diagonal degree node 1024 * 1024
                                                                                                                                                            # same as input features one hot encoding: 32*32
