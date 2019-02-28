# coed to compute weight matrix

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

training_dir ='./Training_data'

# For Keras
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.image_ops_impl import ResizeMethod



# load images, divide into patches and return lost of patch_images and patch_labels
image_data, label_data = remove_unsized_patched(training_dir)

split = 9240                       # which is 30% of 30800 patches

# create Validation set
training_images = image_data[:-split]                   # training data
training_true_class = label_data[:-split]

validation_images = image_data[-split:]
validation_true_class = label_data[-split:]             # validation data

########################## space for normalise data ##################

##########################                          ##################



# The size of the image height and width
image_size = training_images.shape[1]
image_shape = (image_size, image_size)
image_size_flat = image_size * image_size

# Convert linear index to subindices
def ind2sub(linear_index, image_shape):
    i = linear_index % image_shape[0]
    j = math.floor(linear_index / image_shape[0])

    assert (i < image_shape[0])
    assert (j < image_shape[1])

    return i, j


# Convert subindices to linear index
def sub2ind(i, j, image_shape):
    return (j * image_shape[0]) + i

####################################################################### COMPUTE MATRIX IS LIKE A ADJACENCY MATRIX ######################
# Produce a graph Laplacian matrix (and corresponding coordinates) for image, with cardinality `image_shape`
def compute_weight_matrix(image_size_flat, image_shape):
    # Initialise Laplacian of graph matrix
    weight_matrix = np.zeros((image_size_flat, image_size_flat))

    # Populate with edge weights
    # The diagonal is zero for now
    for k in range(0, image_size_flat):
        neighbourhood = np.array([], dtype=np.int32)

        # Subindices for current pixel
        i, j = ind2sub(k, image_shape)

        # Note neighbours
        if i - 1 >= 0:
            neighbourhood = np.append(neighbourhood,
                                      [sub2ind(i - 1, j, image_shape)])
        if i + 1 < image_size:
            neighbourhood = np.append(neighbourhood,
                                      [sub2ind(i + 1, j, image_shape)])
        if j - 1 >= 0:
            neighbourhood = np.append(neighbourhood,
                                      [sub2ind(i, j - 1, image_shape)])
        if j + 1 < image_size:
            neighbourhood = np.append(neighbourhood,
                                      [sub2ind(i, j + 1, image_shape)])

        # Increment neighbours by 1 in weight matrix (Euclidean distance always 1)
        weight_matrix[k, neighbourhood] = 1

    # Create a list of subindices, one for each node in the pre-reduced image
    # These subindices are used for visualisation
    i_s, j_s = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='xy')
    i_s = i_s.reshape(image_size_flat)
    j_s = j_s.reshape(image_size_flat)
    coordinates = np.stack((i_s, j_s))
    coordinates = coordinates.astype(np.float64)

    # Create a diagonal matrix corresponding to the sum of edge weights at each node
    # Then subtract the weight matrix from that diagonal matrix
    # NB: We didn't do this before, as it would have made things difficult when
    # irregularising the data
    # laplacian = np.subtract(np.diag(sum(weight_matrix)), weight_matrix)

    graph_struct = {'weight-matrix': weight_matrix,
                    'coords': coordinates,
                    'indices': np.arange(weight_matrix.shape[0])}

    return graph_struct

unpooled_graph_struct = compute_weight_matrix(image_size_flat, image_shape)


##################################################################### Next is compute features for every node ##################
# here we takes features a set of pixel value,
# Image is RGB image so, it is 3 channel pixel
# each node have a 3 channel pixel value
# here imahe size is (64 * 64), so the total number of nodes requires is 1024, so number of features is 1024*3

image = training_images[0]
print("image shape", image.shape)
feature = np.reshape(image, [64*64,-1])
print("Feature shape: {}".format(feature.shape))

################################################################### Add self loop for identity matri ###############
# this addition of identity matrix show: Addition of nodes + conneceted to other nodes
A = unpooled_graph_struct['weight-matrix']
I = np.matrix(np.eye(A.shape[0]))
A_hat = A + I
print("A_hat")

# mul_A_feature = A_hat * feature
# print("Adding self loop:")

D = np.array(np.sum(A_hat, axis=0))[0]
D_matrix = np.matrix(np.diag(D))
print("D : {}".format(D_matrix))

D_inv = D**-0.5
D_inv_diag = np.diag(D_inv)

A_hat_modified = D_inv_diag * A_hat * D_inv_diag

prod_rule = A_hat_modified * feature

in_unit = feature.shape[1]
out_unit = 16                # number of hidden layer wanted
weights = tf.Variable(tf.random_normal([in_unit,out_unit],stddev=1))

print("D:")



