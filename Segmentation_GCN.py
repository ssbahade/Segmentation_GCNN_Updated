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
from gsp_tensor_ops import graph_conv_block, graph_pooling
from amg_pooling import pool_graph_amg
from pygsp import graphs, filters, plotting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from _datetime import datetime
import helper
from conf_mat import conf_mat


training_dir ='./Training_data'
data_dir = './data'
runs_dir = './runs'

num_classes = 3
IMAGE_SHAPE = (32,32)
epochs = 54                                              # 115 loss will start diverge     # 44 validation started diverge beyond 44
batch_size = 32

# Pooling fields
num_of_pools = 5
num_of_scales = num_of_pools + 1
coarsening_factor = 0.05


# build a graph
def build_graph(x_node):
    print("x_node: {}".format(np.shape(x_node)))
    G = graphs.Grid2d(32,32)                                                           # make sure x[0] having dimension nVertices x nChannel
    G.compute_differential_operator()
    G.compute_fourier_basis()
    G.Utf = tf.convert_to_tensor(G.U, dtype=tf.float32)
    G.plotting['vertex_size'] = 20
    #G.plot()

    '''output_num = 5  # number of output channel like output filters
    weight_num = 1024  # number of nodes
    keep_prob = tf.constant(1.0)
    bnorm_flag = True'''
    return G,G.Utf


# define layer network architecture
def layers(X,num_classes,x_node):
    # To use the GCN data size should be nSample x nVertices x nInChannels
    X = tf.reshape(X,[-1,IMAGE_SHAPE[0]*IMAGE_SHAPE[1],3])
    print("x_node shape: {}".format(x_node.shape))
    x_node = np.reshape(x_node, [-1, num_classes])                                                                       # reshape to nSample x nVertices x nInChannels
    print("After x_node shape: {}".format(x_node.shape))
    G,G.Utf = build_graph(x_node)
    initial_graph = G
    # Pool graph into list of coarsened graphs
    graph_collection = pool_graph_amg(initial_graph, './Results', num_of_scales, coarsening_factor)

    # plot graphs
    for i_graph in range(len(graph_collection)):
        plotting.plot_graph(graph_collection[i_graph])
        plt.show(block=False)
    ################# ######################################################## DOWN ##################################################
    print("X shape: {}".format(X))
    graph_conv1_1 = graph_conv_block(X, graph_collection[0].Utf, graph_collection[0], output_num=32, weight_num=graph_collection[0].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv1_1')                # weight_num = 1024 no. of Nodes, output_num = 32 no. of filters
    print("graph_conv1_1: {}".format(graph_conv1_1))

    graph_conv1_2 = graph_conv_block(graph_conv1_1, graph_collection[0].Utf, graph_collection[0], output_num=64, weight_num=graph_collection[0].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv1_2')
    print("graph_conv1_2: {}".format(graph_conv1_2))
    # pooling 1 ###########################
    pool_1 = graph_pooling(graph_conv1_2,graph_collection[0].R)
    print("pool_1: {}".format(pool_1))



    graph_conv2_1 = graph_conv_block(pool_1, graph_collection[1].Utf, graph_collection[1], output_num=96,weight_num=graph_collection[1].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv2_1')
    print("graph_conv2_1: {}".format(graph_conv2_1))

    graph_conv2_2 = graph_conv_block(graph_conv2_1, graph_collection[1].Utf, graph_collection[1], output_num=128, weight_num=graph_collection[1].N,keep_prob=1.0, bnorm_flag=False, name='graph_conv2_2')
    print("graph_conv2_2: {}".format(graph_conv2_2))
    # pooling 2 ###########################
    pool_2 = graph_pooling(graph_conv2_2, graph_collection[1].R)
    print("pool_2: {}".format(pool_2))



    graph_conv3_1 = graph_conv_block(pool_2, graph_collection[2].Utf, graph_collection[2], output_num=160, weight_num=graph_collection[2].N,keep_prob=1.0, bnorm_flag=False, name='graph_conv3_1')
    print("graph_conv3_1: {}".format(graph_conv3_1))

    graph_conv3_2 = graph_conv_block(graph_conv3_1, graph_collection[2].Utf, graph_collection[2], output_num=192, weight_num=graph_collection[2].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv3_2')
    print("graph_conv3_2: {}".format(graph_conv3_2))

    graph_conv3_3 = graph_conv_block(graph_conv3_2, graph_collection[2].Utf, graph_collection[2], output_num=256, weight_num=graph_collection[2].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv3_3')
    print("graph_conv3_3: {}".format(graph_conv3_3))

    # pooling 3 ###########################
    pool_3 = graph_pooling(graph_conv3_3, graph_collection[2].R)
    print("pool_3: {}".format(pool_3))



    '''graph_conv4_1 = graph_conv_block(pool_3, graph_collection[3].Utf, graph_collection[3], output_num=320, weight_num=graph_collection[3].N,keep_prob=1.0, bnorm_flag=False, name='graph_conv4_1')
    print("graph_conv3_1: {}".format(graph_conv4_1))

    graph_conv4_2 = graph_conv_block(graph_conv4_1, graph_collection[3].Utf, graph_collection[3], output_num=382, weight_num=graph_collection[3].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv4_2')
    print("graph_conv4_2: {}".format(graph_conv4_2))

    graph_conv4_3 = graph_conv_block(graph_conv4_2, graph_collection[3].Utf, graph_collection[3], output_num=512, weight_num=graph_collection[3].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv4_3')
    print("graph_conv3_3: {}".format(graph_conv4_3))

    # pooling 4 ###########################
    pool_4 = graph_pooling(graph_conv4_3, graph_collection[3].R)
    print("pool_3: {}".format(pool_4))



    graph_conv5_1 = graph_conv_block(pool_4, graph_collection[4].Utf, graph_collection[4], output_num=512, weight_num=graph_collection[4].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv5_1')
    print("graph_conv5_1: {}".format(graph_conv5_1))

    graph_conv5_2 = graph_conv_block(graph_conv5_1, graph_collection[4].Utf, graph_collection[4], output_num=512,weight_num=graph_collection[4].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv5_2')
    print("graph_conv5_1: {}".format(graph_conv5_2))

    pool_5 = graph_pooling(graph_conv5_2, graph_collection[4].R)
    print("pool_5: {}".format(pool_5))


    ############################################################### UP ####################################################

    pool_5_up = graph_pooling(pool_5, graph_collection[4].P)
    print("pool_5_up: {}".format(pool_5_up))

    graph_conv5_2_up = graph_conv_block(pool_5_up, graph_collection[4].Utf, graph_collection[4], output_num=512,weight_num=graph_collection[4].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv5_2_up')
    print("graph_conv5_2_up: {}".format(graph_conv5_2_up))

    graph_conv5_1_up = graph_conv_block(graph_conv5_2_up, graph_collection[4].Utf, graph_collection[4], output_num=512,weight_num=graph_collection[4].N, keep_prob=1.0, bnorm_flag=False,name='graph_conv5_1_up')
    print("graph_conv5_1: {}".format(graph_conv5_1_up))



    pool_4_up = graph_pooling(graph_conv5_1_up, graph_collection[3].P)
    print("pool_4_up: {}".format(pool_4_up))

    # First convolutional block
    graph_conv4_3_up = graph_conv_block(pool_4_up, graph_collection[3].Utf, graph_collection[3], output_num=512,weight_num=graph_collection[3].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv4_3_up')
    print("graph_conv4_3_up: {}".format(graph_conv4_3_up))

    graph_conv4_2_up = graph_conv_block(graph_conv4_3_up, graph_collection[3].Utf, graph_collection[3], output_num=382,weight_num=graph_collection[3].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv4_2_up')
    print("graph_conv4_2_up: {}".format(graph_conv4_2_up))

    graph_conv4_1_up = graph_conv_block(graph_conv4_2_up, graph_collection[3].Utf, graph_collection[3], output_num=320,weight_num=graph_collection[3].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv4_1_up')
    print("graph_conv4_1_up: {}".format(graph_conv4_1_up))'''



    # First upsampling
    pool_3_up = graph_pooling(pool_3, graph_collection[2].P)
    print("pool_3_up: {}".format(pool_3_up))

    # First convolutional block
    graph_conv3_3_up = graph_conv_block(pool_3_up, graph_collection[2].Utf, graph_collection[2], output_num=256,weight_num=graph_collection[2].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv3_3_up')
    print("graph_conv3_3_up: {}".format(graph_conv3_3_up))

    graph_conv3_2_up = graph_conv_block(graph_conv3_3_up, graph_collection[2].Utf, graph_collection[2], output_num=192,weight_num=graph_collection[2].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv3_2_up')
    print("graph_conv3_2_up: {}".format(graph_conv3_2_up))

    graph_conv3_1_up = graph_conv_block(graph_conv3_2_up, graph_collection[2].Utf, graph_collection[2], output_num=160,weight_num=graph_collection[2].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv3_1_up')
    print("graph_conv3_1_up: {}".format(graph_conv3_1_up))



    # Second upsampling
    pool_2_up = graph_pooling(pool_2, graph_collection[1].P)
    print("pool_2_up: {}".format(pool_2_up))

    graph_conv2_2_up = graph_conv_block(pool_2_up, graph_collection[1].Utf, graph_collection[1], output_num=128,weight_num=graph_collection[1].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv2_2_up')
    print("graph_conv2_2_up: {}".format(graph_conv2_2_up))

    graph_conv2_1_up = graph_conv_block(graph_conv2_2, graph_collection[1].Utf, graph_collection[1], output_num=96,weight_num=graph_collection[1].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv2_1_up')
    print("ggraph_conv2_1_up: {}".format(graph_conv2_1_up))



    # Third upsampling
    pool_1_up = graph_pooling(graph_conv2_1_up, graph_collection[0].P)
    print("pool_1_up: {}".format(pool_1_up))

    graph_conv1_2_up = graph_conv_block(pool_1_up, graph_collection[0].Utf, graph_collection[0], output_num=64,weight_num=graph_collection[0].N, keep_prob=1.0, bnorm_flag=False, name='graph_conv1_2_up')
    print("graph_conv1_2_up: {}".format(graph_conv1_2_up))

    graph_conv1_1_up = graph_conv_block(graph_conv1_2_up, graph_collection[0].Utf, graph_collection[0], output_num=32, weight_num=graph_collection[0].N,keep_prob=1.0, bnorm_flag=False,name='graph_conv1_1_up')  # weight_num = 1024 no. of Nodes, output_num = 32 no. of filters
    print("graph_conv1_1_up: {}".format(graph_conv1_1_up))

    graph_conv_0 = graph_conv_block(graph_conv1_1_up, graph_collection[0].Utf, graph_collection[0], output_num=3, weight_num=graph_collection[0].N,keep_prob=1.0, bnorm_flag=False, name='graph_conv_0')
    print("graph_conv_0: {}".format(graph_conv_0))


    print("End layer function:")

    ''''# layer 1 ########################
    conv1_1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv1_1")
    print("conv1_1: {}".format(conv1_1))
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv1_2" )
    print("conv1_2: {}".format(conv1_2))
    pool_1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2 , name="pool_1")
    print("pool_1: {}".format(pool_1))

    # layer 2
    conv2_1 = tf.layers.conv2d(inputs=pool_1, filters=96, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv2_1")
    print("conv2_1: {}".format(conv2_1))
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv2_2")
    print("conv2_2: {}".format(conv2_2))
    pool_2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2 , name="pool_2")
    print("pool_2: {}".format(pool_2))

    # layer 3
    conv3_1 = tf.layers.conv2d(inputs=pool_2, filters=160, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv3_1")
    print("conv3_1: {}".format(conv3_1))
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=192, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv3_2")
    print("conv3_2: {}".format(conv3_2))
    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv3_3")
    print("conv3_3: {}".format(conv3_3))
    pool_3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2 , name="pool_3")
    print("pool_3: {}".format(pool_3))

    # layer 4
    conv4_1 = tf.layers.conv2d(inputs=pool_3, filters=320, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv4_1")
    print("conv4_1: {}".format(conv4_1))
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv4_2")
    print("conv4_2: {}".format(conv4_2))
    conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="conv4_3")
    print("conv4_3: {}".format(conv4_3))
    pool_4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2, name="pool_4")
    print("pool_4: {}".format(pool_4))

    # layer 5
    conv5_1 = tf.layers.conv2d(inputs=pool_4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv5_1")
    print("conv5_1: {}".format(conv5_1))
    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv5_2")
    print("conv5_2: {}".format(conv5_2))
    conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv5_3")
    print("conv5_3: {}".format(conv5_3))
    pool_5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2 ,name="pool_5")
    print("pool_5: {}".format(pool_5))

    # flat layer
    # shape_flat = int(np.prod(pool_5.get_shape()[1:]))
    # layer_flat = tf.reshape(pool_5,[-1,shape_flat])

    fc6 = tf.layers.conv2d(inputs=pool_5, filters=4096, kernel_size=[7, 7], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="fc6")
    print("fc6: {}".format(fc6))
    fc7 = tf.layers.conv2d(inputs=fc6, filters=4096, kernel_size=[1, 1], padding="same", activation=tf.nn.relu ,reuse=tf.AUTO_REUSE, name="fc7")
    print("fc7: {}".format(fc7))

    print("fc7 shape : {}".format(fc7.get_shape()))

    # ############################################################## UP #####################################

    # Apply 1x1 convolution in place of fully connected layer
    fc8 = tf.layers.conv2d(fc7, filters=num_classes, kernel_size=1, reuse=tf.AUTO_REUSE, name="fc8")  # num_classes = 3
    print("fc8 shape :{}".format(fc8))

    print("pool_4.get_shape().as_list()[-1] : ", pool_4.get_shape().as_list()[-1])

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fc9 = tf.layers.conv2d_transpose(fc8, filters=pool_4.get_shape().as_list()[-1], kernel_size=4, strides=2,
                                     padding='SAME', reuse=tf.AUTO_REUSE, name="fcn9")
    print("fc9: {}".format(fc9))

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fc9, pool_4,  name="fcn9_plus_vgg_layer4")
    print("fcn9_skip_connected: {}".format(fcn9_skip_connected))

    # Upsample again
    fc10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=pool_3.get_shape().as_list()[-1], kernel_size=4,
                                      strides=2, padding='SAME', reuse=tf.AUTO_REUSE, name="fcn10_conv2d")
    print("fc10: {}".format(fc10))

    # Add skip connection
    fc10_skip_connected = tf.add(fc10, pool_3,  name="fcn10_plus_vgg_layer3")
    print("fc10_skip_connected: {}".format(fc10_skip_connected))

    # Upsample again
    fc11 = tf.layers.conv2d_transpose(fc10_skip_connected, filters=num_classes, kernel_size=16, strides=(8, 8),
                                      padding='SAME', reuse=tf.AUTO_REUSE, name="fcn11")
    print("fc11 : {}".format(fc11))'''
    return graph_conv_0


def optimize(nn_last_layer, correct_label, learning_rate, NUMBER_OF_CLASSES):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, NUMBER_OF_CLASSES), name="fcn_logits")
    correct_label_reshaped = tf.cast(tf.reshape(correct_label, [-1]),dtype=tf.int32)
    # print("corr labe flatten shape :",correct_label_reshaped.get_shape())

    correct_label_reshaped_one_hot = tf.one_hot(correct_label_reshaped, NUMBER_OF_CLASSES)  # one hot encoding

    print("logits: {} and label: {} ".format(logits,correct_label_reshaped_one_hot))
    print("Executing optimize")

    # Confusion matrix
    # conf_mat = tf.confusion_matrix(labels=correct_label_reshaped_one_hot,predictions=logits,num_classes=3)

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label_reshaped_one_hot[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")


    # Accuracy
    #acc, acc_op = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(correct_label_reshaped_one_hot, 1), predictions=tf.argmax(logits, 1),num_classes=num_classes)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped_one_hot, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    Training_accuracy_summary = tf.summary.scalar('Accuracy_wrt_Batch_summary', acc)
    Training_loss_summary = tf.summary.scalar('Loss_wrt_Batch_summary', loss_op)
    Testing_accuracy_summary = tf.summary.scalar('Testing_Acc_summary', acc)

    # Training summary
    merged_summaries = tf.summary.merge_all()

    # Precision and Recall
    #prec, prec_op = tf.metrics.precision_at_k(labels=tf.argmax(correct_label_reshaped_one_hot, 1), predictions=tf.argmax(logits, 1),k=3,class_id=0)


    return logits, train_op,loss_op, acc, correct_label_reshaped_one_hot, merged_summaries

def train_nn(sess,cross_entropy_loss,logits,acc, train_op,get_batches_fn,X,correct_label,keep_prob,learning_rate, write, Val_write,merged_summ):
    keep_prob_value = 0.23
    learning_rate_value = 0.0001
    t1=0
    t2=0
    list_test_batch = []
    list_test_gtBatch = []
    cf_train_gt_list = []
    cf_train_pred_list = []
    cf_val_pred_list = []

    list_train_acc = 0
    list_test_acc = 0
    train_acc = 0
    test_acc = 0
    train_loss = 0
    val_loss = 0


    for X_batch, gt_batch, test_batch, test_gt_batch, len_xtrain, len_xtest, num_epoch in get_batches_fn(batch_size,
                                                                                                         epochs, sess):
        if test_batch.shape != (0,):
            list_test_batch.append(test_batch)
            list_test_gtBatch.append(test_gt_batch)
        cf_train_gt_list.append(gt_batch)
        _,loss_train_batch,acc_batch, train_batch_summ, cf_train_pred = sess.run([train_op,cross_entropy_loss,acc, merged_summ, logits],feed_dict={X: X_batch, correct_label: gt_batch,
                                      keep_prob: keep_prob_value, learning_rate: learning_rate_value})
        cf_train_pred_list.append(cf_train_pred)
        write.add_summary(train_batch_summ, t1)

        train_loss += loss_train_batch
        # loss_train_batch_summ = tf.summary.scalar('loss_train_batch_summary',loss_train_batch)
        # acc_batch_summ = tf.summary.scalar('acc_batch_summary', acc_batch)
        list_train_acc += acc_batch
        #print("Train Batch Accuracy acc : {} and acc_op : {} .................... Loss : {} ".format(acc_batch,acc_op_batch,loss_train_batch))
        t1 = t1 + 1
        if len_xtrain == t1:
            train_acc = list_train_acc / len_xtrain
            tr_loss = train_loss/len_xtrain
            print("Epoch {}: Training Accuracy = {} ...................... Loss = {}".format(num_epoch, train_acc, tr_loss))

            list_train_acc = 0
            loss_test_batch = 0
            train_loss = 0

            for g in range(0,len(list_test_batch)):
                loss_test_batch, acc_test_batch,val_batch_summ, cf_test_pred = sess.run([cross_entropy_loss, acc,merged_summ, logits],feed_dict={X: list_test_batch[g], correct_label: list_test_gtBatch[g]})
                list_test_acc += acc_test_batch
                cf_val_pred_list.append(cf_test_pred)
                val_loss += loss_test_batch
                Val_write.add_summary(val_batch_summ, g)
            test_acc = list_test_acc / len(list_test_batch)
            vl_loss = val_loss / len(list_test_batch)
            val_loss = 0
            print("Epoch {}: Val Accuracy = {}  ............validation Loss = {}".format(num_epoch, test_acc, vl_loss))
            # Validation_accuracy_summary = tf.summary.scalar('Validation_acc_summary', test_acc)
            # Validation_loss_summary = tf.summary.scalar('Validation_loss_summary', vl_loss)
            CF_TRAIN_GT_ = cf_train_gt_list.copy()
            CF_TRAIN_PRED_ = cf_train_pred_list.copy()
            CF_VAL_PRED_ = cf_val_pred_list.copy()
            CF_VAL_GT_ = list_test_gtBatch.copy()

            cf_train_gt_list.clear()                    # clear cf_train_gt for aim to store last epoch  train_gt value
            cf_train_pred_list.clear()                  # clear cf_train_pred for aim to store last epoch logit train_pred value
            cf_val_pred_list.clear()
            #Validation summary
            #Validation_summaries = tf.summary.merge([test_acc, vl_loss])


            list_test_acc = 0
            list_test_batch.clear()
            list_test_gtBatch.clear()
            t1 = 0

        '''if test_batch.shape != (0,):
            loss_test_batch, acc_test_batch, acc_op_test_batch = sess.run(
                [cross_entropy_loss, acc, acc_op],
                feed_dict={X: test_batch, correct_label: test_gt_batch,
                           keep_prob: 1.0})
            #print("Val Batch Accuracy acc : {} and acc_op : {} .................... Loss : {} ".format(acc_test_batch, acc_op_test_batch,
                                                                                                   #loss_test_batch))
        t2=t2+1
        if len_xtest == t2:
            print("Epoch {}: Test Accuracy = {} Loss = {}".format(num_epoch, acc_test_batch, loss_test_batch))
            t2 = 0'''
    return train_acc, test_acc, tr_loss, vl_loss, CF_TRAIN_PRED_,CF_VAL_PRED_,CF_TRAIN_GT_,CF_VAL_GT_




'''
# The size of the image height and width
image_size = training_images.shape[1]
image_shape = (image_size, image_size)
image_size_flat = image_size * image_size

New = training_images[0]
New_reshaped = np.reshape(New,(New.shape[0]*New.shape[1],-1))


G = graphs.NNGraph(New_reshaped)
G.compute_differential_operator()
G.compute_fourier_basis()
G.Utf = tf.convert_to_tensor(G.U, dtype=tf.float32)
G.plotting['vertex_size'] = 20
G.plot()

output_num = 5                                       # number of output channel like output filters
weight_num = 1024                                    # number of nodes
keep_prob = tf.constant(1.0)
bnorm_flag = True


#######################################  create a graph convolutional layers

input_data = tf.placeholder(tf.float32, shape=[None, training_images.shape[1], training_images.shape[2], training_images.shape[3]])
labels = tf.placeholder(tf.float32,shape=[None, training_images.shape[1], training_images.shape[2]])


# first layer
graph_conv = graph_conv_block(input_data, G.Utf, G, output_num=30, weight_num=1024, keep_prob=1.0, bnorm_flag=False,name='layer1')
# second layer
graph_conv_2 = graph_conv_block(graph_conv, G.Utf, G, output_num=10, weight_num=5, keep_prob=1.0, bnorm_flag=False,name='layer2')

with tf.name_scope('loss'):
    true_class_reshape = tf.reshape(labels, shape=[-1,training_images.shape[1]*training_images.shape[2]])
    true_class_cast = tf.cast(true_class_reshape, tf.int32)
    true_one_hot = tf.one_hot(true_class_cast, 3)
    log = tf.argmax(tf.nn.softmax(graph_conv_2, axis=0), axis=1)
    lab = tf.argmax(true_one_hot, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph_conv_2, labels=true_one_hot))
    print("logit shape: {} and label shape: {}".format(graph_conv_2.get_shape(), true_one_hot.get_shape()))
    train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(loss)
'''

def run():


    #fold = 1
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # called function to divide image image into patches
    image_data, label_data = remove_unsized_patched(training_dir)
    print("image_data: {} and label_data: {}".format(image_data.shape, label_data.shape))

    # load images, divide into patches and return lost of patch_images and patch_labels
    # image_data = image_data.astype(np.float)
    # label_data = label_data.astype(np.float)
    n_splits = 5
    kf = KFold(n_splits)
    fold = 1
    for train, test in kf.split(image_data):
        x_train, x_test = image_data[train], image_data[test]
        y_train, y_test = label_data[train], label_data[test]

        print("FOLD: {}".format(fold))
    #X_train, X_test, y_train, y_test = train_test_split(image_data, label_data, test_size = 0.33)

        X = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[0], num_classes], name="input_x")  # define placeholder x for 4d input value to input
        correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1]], name="Label")
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)

        get_batches_fn = helper.gen_batch_function(x_train, x_test, y_train, y_test, IMAGE_SHAPE)

        with tf.Session() as session:
            print("in tf.session")
            model_output = layers(X, num_classes,x_train[0])
            logits, train_op, cross_entropy_loss, acc, one_hot, merged_summaries_ = optimize(model_output, correct_label, learning_rate, num_classes)
            # Initialize all variables
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            Train_write = tf.summary.FileWriter("./data/add_num/Train", session.graph)
            Val_write = tf.summary.FileWriter("./data/add_num/Validation", session.graph)
            Test_write = tf.summary.FileWriter("./data/add_num/Test", session.graph)

            print("Model build successful, starting training")

            tr, tst, train_loss, test_loss, cf_train_pre_list, cf_val__pred_list, cf_train_gt, cf_val_gt = train_nn(session,cross_entropy_loss,logits,acc, train_op,get_batches_fn,X,correct_label,keep_prob,learning_rate, Train_write, Val_write, merged_summaries_)
            print(" Validation {} : Train Accuracy = {}, Test Accuracy = {}, Train Loss = {}, Test Loss = {}".format(fold, tr, tst, train_loss, test_loss))

            saver = tf.train.Saver()
            # save the variable in the disk
            saved_path = saver.save(session, './saved_variable/model.ckpt' + str(fold))
            print('model saved in {}'.format(saved_path))

            # Run the model with the test images and save
            cf_test_p, cf_test_g = helper.save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, X, cross_entropy_loss, acc,  correct_label, Test_write, merged_summaries_)
            conf_mat(cf_train_pre_list, cf_train_gt, cf_val__pred_list, cf_val_gt, cf_test_p, cf_test_g)
            # print("Tensorboard graph :")
        tf.reset_default_graph()
        fold += 1
        print("Validation Execution time : {}".format(datetime.now()))
    print("End Time : {}".format(datetime.now()))
    print("Execution time : {}".format(datetime.now() - start))
    print("ALL DONE !")





#--------------------------
# MAIN
#--------------------------
if __name__ == '__main__':
    start = datetime.now()
    print("Start Time",start)
    run()
