import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import numpy as np
from scipy import interpolate

model_precision = tf.float32

#From mdeff
def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.
    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis

# From mdeff
def spline(G, W, K, Fin, Fout):
    """Spline interpolation for Graph-CNN weights using pygsp Graph

        Args:
            G: pygsp Graph object, with fourier basis computed
            W: tensor of weights to be interpolated, with shape K x Fout * Fin
    """
    # Sizes
    nNode = G.N
    lamb = G.e

    # Spline basis
    B = bspline_basis(K, lamb, degree=3)  # M x K
    # B = bspline_basis(K, len(lamb), degree=3)  # M x K
    B = tf.constant(B, dtype=model_precision)

    # Weights
    W = tf.matmul(B, W)  # M x Fout*Fin
    W = tf.reshape(W, [nNode, Fout, Fin])
    W = tf.transpose(W,perm=[0,2,1])
    return W

# From mdeff
def filter_in_fourier(x, G, W):
    # TODO: N x F x M would avoid the permutations
    data_shape = tf.shape(x)
    nSamp = data_shape[0:1]
    nNode = data_shape[1:2]
    Fin = data_shape[2:3]
    W = tf.transpose(W,perm=[0,2,1])
    Fout = tf.shape(W)[1:2]
    #nSamp, nNode, Fin = int(nSamp), int(nNode), int(Fin)
    x = tf.transpose(x, perm=[1, 2, 0])  # nNode x Fin x nSamp
    # Transform to Fourier domain
    x = tf.reshape(x, tf.concat([nNode, Fin * nSamp],axis=0))  # nNode x Fin*nSamp
    x = tf.matmul(G.Utf, x)  # nNode x Fin*nSamp
    x = tf.reshape(x, tf.concat([nNode, Fin, nSamp],axis=0))  # nNode x Fin x nSamp
    # Filter
    x = tf.matmul(W, x)  # for each feature
    x = tf.transpose(x)  # nSamp x Fout x nNode
    x = tf.reshape(x, tf.concat([nSamp * Fout, nNode],axis=0))  # nSamp*Fout x nNode
    # Transform back to graph domain
    x = tf.matmul(x, G.Utf)  # nSamp*Fout x nNode
    x = tf.reshape(x, tf.concat([nSamp, Fout, nNode],axis=0))  # nSamp x Fout x nNode
    return tf.transpose(x, perm=[0, 2, 1])  # nSamp x M x Fout

def tf_gft(s, U):
    """Tensor based GFT.

        Args:
            U: tensor of size nV by nV, Eigenvector matrix of current GSP graph (G.U)
            s: tensor of size nS by nV by nC, stack of graph signals in spatial domain

        Returns:
            s_hat: tensor of size nV by nC by nS, stack of graph signals in frequency domain

        Example: TODO
        """
# if type == 'innerprod':
#     with tf.device('/cpu:0'):
#         shape = s.get_shape().as_list()
#         s = tf.expand_dims(s, axis=3)
#         s = tf.transpose(s, perm=[0, 2, 1, 3])
#         s = tf.reshape(s, shape=[-1, shape[1], 1])
#         s_hat = tf.reduce_sum(tf.multiply(U, s), 1, keepdims=True)
#         s_hat = tf.transpose(tf.reshape(s_hat, shape=[-1, shape[2], shape[1]]), perm=[0, 2, 1])

# elif type == 'tensordot':
#     s = tf.transpose(s, perm=[1, 2, 0])
#     s_hat = tf.tensordot(tf.conj(U), s, ([0], [0]))
#     s_hat = tf.transpose(s_hat, perm=[2, 0, 1])
#
# elif type == 'einsum':
    s_hat = tf.einsum('ij,aik->ajk', tf.conj(U), s)

    return s_hat


def filtering_checker(s,f,G):
    U = tf.constant(G.U.astype(np.float32))
    s_hat = tf_gft(s, U)
    f = tf.cast(tf.image.resize_images(f, [s.shape[1], 1], method=ResizeMethod.BICUBIC), model_precision)
    interpolated_weights = tf.reshape(f, [s.shape[1], f.shape[1], f.shape[2]])  # weights wants to be nV by nI by nO

    s_hat_filtered = tf.reduce_sum(
                tf.multiply(tf.expand_dims(s_hat, axis=3), f), axis=2)
    # s_hat_filtered = tf.einsum('svc,vio->svo', s_hat, f)
    return s_hat_filtered


def gft_checker(s, G):
    U = tf.constant(G.U.astype(np.float32))
    gsp = G.gft(np.squeeze(s.eval()))
    mmul = tf_gft(s, U, 'innerprod')
    tdot = tf_gft(s, U, 'tensordot')
    esum = tf_gft(s, U, 'einsum')

    print(np.all((np.reshape(mmul.eval(),[-1]) - np.reshape(gsp,[-1])) < 1e-10))
    print(np.all((np.reshape(tdot.eval(),[-1]) - np.reshape(gsp,[-1])) < 1e-10))
    print(np.all((np.reshape(esum.eval(),[-1]) - np.reshape(gsp,[-1])) < 1e-10))


    gspinv = G.igft(gsp)
    mmulinv = tf_igft(mmul, U, 'innerprod')
    tdotinv = tf_igft(tdot, U, 'tensordot')
    esuminv = tf_igft(esum, U, 'einsum')

    print(np.all((np.reshape(mmulinv.eval(),[-1]) - np.reshape(gspinv,[-1])) < 1e-10))
    print(np.all((np.reshape(tdotinv.eval(),[-1]) - np.reshape(gspinv,[-1])) < 1e-10))
    print(np.all((np.reshape(esuminv.eval(),[-1]) - np.reshape(gspinv,[-1])) < 1e-10))

    print(np.all((np.reshape(gspinv,[-1]) - np.reshape(s.eval(),[-1])) < 1e-10))
    print(np.all((np.reshape(mmulinv.eval(),[-1]) - np.reshape(s.eval(),[-1])) < 1e-10))
    print(np.all((np.reshape(tdotinv.eval(),[-1]) - np.reshape(s.eval(),[-1])) < 1e-10))
    print(np.all((np.reshape(esuminv.eval(),[-1]) - np.reshape(s.eval(),[-1])) < 1e-10))
    return gspinv, esuminv.eval(), esuminv.eval(), esuminv.eval()

def tf_igft(s_hat, U):
    """Tensor based Inverse GFT.

        Args:
            U: tensor of size nV by nV, Eigenvector matrix of current GSP graph (G.U)
            s: tensor of size nS by nV by nC, stack of graph signals in frequency domain

        Returns:
            s_hat: tensor of size nS by nV by nC, stack of graph signals in spatial domain

        Example: TODO
        """
    # with tf.device('/cpu:0'):
    U = tf.transpose(U)
    # if type == 'innerprod':
#     shape = s_hat.get_shape().as_list()
#     s_hat = tf.expand_dims(s_hat, axis=3)
#     s_hat = tf.transpose(s_hat, perm=[0, 2, 1, 3])
#     s_hat = tf.reshape(s_hat, shape=[-1, shape[1], 1])
#     s = tf.reduce_sum(tf.multiply(U, s_hat), 1, keepdims=True)
#     s = tf.transpose(tf.reshape(s, shape=[-1 ,shape[2], shape[1]]), perm=[0, 2, 1])

# elif type == 'tensordot':
#     s_hat = tf.transpose(s_hat, perm=[1, 2, 0])
#     s = tf.tensordot(tf.conj(U), s_hat, ([0], [0]))
#     s = tf.transpose(s, perm=[2, 0, 1])
#
# elif type == 'einsum':
    s = tf.einsum('ij,aik->ajk', tf.conj(U), s_hat)

    return s


def new_biases(length):
    return tf.Variable(tf.zeros([length],dtype=model_precision)+0.1,name='bias', dtype=model_precision)


def graph_conv_layer(data,
                     fourier_base, G,
                     output_num,
                     tracked_weights,
                     filt_type='fourier'):
    """Convolution of graph signal, currently implemented using the smooth spectral multiplier filtering method.

    Args:
        data: `Tensor`, input batch of graph signals in spatial domain, size: nSamples x nVertices x nInChannels.
        fourier_base: `Tensor`, fourier basis of the graph, oriented columnwise, size: nVertices x nVertices.
        output_num: `int`, scalar value of number of output feature maps, nOutChannels.
        filt_type: `str`, string switch for type of graph convolution method used. Options: fourier
        dropout: `bool`, boolean flag denoting use of dropout on the current convolutional layer.

    Returns:
        `Tensor` of shape nSamples x nVertices x nOutChannels, batch of filtered graph signals in spatial domain.

    Example: TODO
    """

    # Data shape: nSamples x nVertices x NChannels
    with tf.variable_scope('signal_convolution_' + str(output_num)):
        if filt_type == 'fourier':
            # Make weights
            input_num = data.shape.as_list()[-1]

            # Make biases
            biases = new_biases(output_num)

            # GFT
            #spectral_data = tf_gft(data, fourier_base)

            # A smooth filter
            weights = tf.get_variable('weights',
                                      shape=[tracked_weights, output_num * input_num],
                                      initializer=tf.truncated_normal_initializer(0,0.1),
                                     #constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
                                      dtype=model_precision)

            interpolated_weights = spline(G,weights,K=tracked_weights, Fin=input_num, Fout=output_num)


            # Apply spectral filtering
            #filtered_spectral_data = tf.reduce_sum(
            #    tf.multiply(tf.expand_dims(spectral_data, axis=3), interpolated_weights), axis=2)

            # Reverse Fourier transformation
            #output_data = tf_igft(filtered_spectral_data, fourier_base)

            # mdeff all in one
            output_data = filter_in_fourier(data,G,interpolated_weights)

            # Add bias
            output_data = tf.nn.bias_add(output_data, biases)

        return output_data


def graph_conv_block(data,
                     fourier_base,G,
                     output_num,
                     weight_num,
                     keep_prob,
                     bnorm_flag,
                     filt_type='fourier',
                     activation=tf.nn.leaky_relu,
                     name=None,
                     reuse=False):
    scopename = 'sigconv_block'
    if name is not None:
        scopename += '_' + name

    with tf.variable_scope(scopename, reuse=reuse):
        # Filter
        output_data = graph_conv_layer(data, fourier_base, G, output_num, weight_num, filt_type)

        # Batch norm
        # output_data = tf.layers.batch_normalization(output_data, training=bnorm_flag)
        output_data = tf.expand_dims(output_data, axis=2)
        output_data = tf.contrib.layers.batch_norm(output_data, is_training=bnorm_flag)
        output_data = tf.squeeze(output_data, axis=2)

        # Activation
        output_data = activation(output_data)#, alpha=tf.cast(0.2,dtype=model_precision))
        # Dropout
        output_data = tf.layers.dropout(output_data, keep_prob, noise_shape=[None,1,output_data.get_shape()[2]])

        return output_data


# ### Pooling Operation
def graph_pooling(sig, projection, name=None):
    """Pooling using AMG kernel interpolation

        Args:
            sig: `Tensor`, input batch of graph signals in spatial domain, size: nSamples x nVertices x nInChannels.
            projection: `ndarray`, AMG interp matrix,
                    if restriction matrix (fine to coarse) size: nFineVertice x nCoarseVertices.
                    if projection matrix (coarse to fine) size: nCoarseVertices x nFineVertice.

        Returns:
            `Tensor` of shape nSamples x nVertices x nOutChannels, batch of coarsened or refined signals.

        Types:
            MatMul:


            TensorDot:
        # sig = tf.transpose(sig, perm=[1, 2, 0])  # turn sig from nS by nV by nC -> nV by nC by nS
        # pool_sig = tf.tensordot(projection, sig, ([1], [0]))
        # pool_sig = tf.transpose(pool_sig, perm=[2, 0, 1])

            Einsum:
        # projection = tf.transpose(projection)
        # pool_sig = tf.einsum('ij,aik->ajk', projection, sig)

        Example: TODO
        """
    scope_name = 'graph_pooling'
    if not name is None:
        scope_name += '_' + name
    with tf.variable_scope('graph_pooling'):


        projection = tf.transpose(projection)
        pool_sig = tf.einsum('ij,aik->ajk', projection, sig)
        return pool_sig
