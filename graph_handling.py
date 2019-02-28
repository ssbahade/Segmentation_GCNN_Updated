import pickle
import tensorflow as tf
from pygsp import graphs
import numpy as np

model_precision = tf.float32

def save_GSP_graph(G, path):
    GSP_params = {'W': G.W, 'coords': G.coords}

    if hasattr(G,'R'):
        GSP_params['R'] = G.R

    if hasattr(G,'P'):
        GSP_params['P'] = G.P

    with open(path, 'wb') as output:
        pickle.dump(GSP_params, output, pickle.HIGHEST_PROTOCOL)

def load_GSP_graph(path):
    G_params = pickle.load(open(path,'rb'))
    G = graphs.Graph(G_params['W'])
    G.compute_fourier_basis()
    G.Utf = tf.convert_to_tensor(G.U, dtype=model_precision)
    G.coords = G_params['coords']
    G.plotting['vertex_size'] = 20

    if 'R' in G_params:
        G.R = tf.convert_to_tensor(G_params['R'], dtype=model_precision)
    if 'P' in G_params:
        G.P = tf.convert_to_tensor(G_params['P'], dtype=model_precision)

    return G
    # Load in pickled data