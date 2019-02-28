import numpy as np
import scipy as sps
from os import path
from pygsp import graphs
import tensorflow as tf
from graph_handling import save_GSP_graph, load_GSP_graph

model_precision = tf.float32

# # Agglomerative Pooling of Graphs
#
# ### An adaption of the AGM method given by ['shaibagon'](https://github.com/shaibagon/amg).
#
# ### The graphs are described by a structure:
# #### * `'laplacian'`—The Laplacian matrixof the graph.
# #### * `'coords'`—The coordinates of each node in the graph.
# #### * `'projection'`—The projection function to go from the current layer to the next layer.
# #### * `'restriction'`—The restriction function to go from the current layer to the previous layer.
# #### * `'node-indices'`—The node indices for each row/column in the Laplacian matrix.


def greedy_coarsen(normalised_weight_matrix, order_of_consideration, coarsening_factor):
    number_of_nodes = normalised_weight_matrix.shape[0]

    # This is where we say we keep nodes
    nodes_to_keep_binary_array = np.zeros(number_of_nodes, dtype=np.bool)

    # List of sums of the column for each node
    sums_of_columns_considered = np.zeros(number_of_nodes)

    for current_node_of_focus in order_of_consideration:
        assert (current_node_of_focus > -1 and current_node_of_focus < number_of_nodes)

        if sums_of_columns_considered[current_node_of_focus] <= coarsening_factor:
            nodes_to_keep_binary_array[current_node_of_focus] = True

            for column in range(number_of_nodes):
                sums_of_columns_considered[column] += normalised_weight_matrix[current_node_of_focus, column]

    return nodes_to_keep_binary_array


# See the function of the same name in shaibagon's AMG repo
def fine2coarse(weight_matrix, coarsening_factor):
    weight_matrix = sps.sparse.csr_matrix(weight_matrix)
    number_of_nodes = weight_matrix.shape[0]

    # Only used for greedy coarsening
    w = weight_matrix
    w2 = np.asarray(np.divide(1, w.sum(axis=1)))

    normalised_weight_matrix = sparseMTimesD(w, w2)

    # A random ordering of the nodes
    order_of_consideration = np.random.permutation(number_of_nodes)

    # Returns a binary array
    c = greedy_coarsen(normalised_weight_matrix, order_of_consideration, coarsening_factor)
    ci = np.where(c)[0]
    # 'Compute the interp matrix'
    projection = weight_matrix[:, ci]
    psum = np.asarray(projection.sum(axis=1))
    projection = np.asarray(sparseMTimesD(projection, 1 / (psum + np.finfo(float).eps)))

    # The following lines work like [x, y, val] = find(_) in MATLAB
    ii, jj = np.nonzero(projection)  # returns a tuple
    pji = projection[ii, jj]

    # Select the points not included in the coarsening
    sel = ~c[ii]

    # Row concatenation of two row vectors
    row_cat = lambda x, y: np.concatenate((x, y))

    # A sparse matrix for the projections
    # Takes a 2-element tuple of values and a 2-element tuple, given by columns_and_rows_where_true
    projection = sps.sparse.csr_matrix(
        (row_cat(pji[sel], np.ones((np.sum(c)))), (row_cat(jj[sel], np.arange(np.sum(c))), row_cat(ii[sel], ci))),
        shape=(projection.shape[1], projection.shape[0])).T

    return c, projection


def sparseMTimesD(m, d):
    c = np.matmul(np.diag(d.flatten()), m.todense())
    return c


def amg(graph, number_of_scales, coarsening_factor):
    """Aglomerative Multi-Grid Pooling.

    Args:
        graph: `PyGSP Graph object`, input graph to pool.
        number_of_scales: `int`, Number of pooled graphs to create.

    Returns:
        List of weight matrices, each succesively pooled using AMG.
        List of projection matrices.
        List of restriction matrices.
        List of coarsened nodes.

    Example: TODO
    """
    number_of_nodes = graph.N

    # A list of the indices marking the nodes in the finest clustering
    fine_node_indices = np.arange(number_of_nodes)

    projections = []
    restrictions = []
    coarsened_nodes = []
    weight_matrices = []

    weight_matrices.append(graph.W)

    for scale_num in range(number_of_scales):
        c, p = fine2coarse(weight_matrices[scale_num], coarsening_factor)
        projections.append(p)
        coarsened_nodes.append(fine_node_indices[c])

        if scale_num < number_of_scales - 1:
            weight_matrices.append(projections[scale_num].T * weight_matrices[scale_num] * projections[scale_num])
            weight_matrices[-1].setdiag(0)
            fine_node_indices = coarsened_nodes[scale_num]

    # Calculate inverse of projection
    restrictions = [
        np.asarray(sparseMTimesD(projection.T, np.asarray(np.divide(1, projection.sum(axis=0)))))
        for projection in projections
    ]
    projections = [
        projection.todense()
        for projection in projections
    ]

    return weight_matrices, projections, restrictions, coarsened_nodes


def pool_graph_amg(G, filename_for_run, num_of_scales, coarsening_factor):
    # Load previous poolings if they exist
        weight_list, projections_list, restrictions_list, coords_list = amg(G, num_of_scales, coarsening_factor)

        # Create stack of pooled GSP graphs and add in the projection and restriction functionality
        G_list = []

        if not path.exists(path.join(filename_for_run, 'GSP_graph_params*.pkl')):
            G_list.append(G)
            G_list[0].P = projections_list[0]
            G_list[0].R = restrictions_list[0]

            for iG in range(1, num_of_scales):
                G_list.append(graphs.Graph(weight_list[iG]))
                G_list[iG].coords = G_list[0].coords[coords_list[iG - 1], :]
                G_list[iG].compute_fourier_basis()
                G_list[iG].P = projections_list[iG]
                G_list[iG].R = restrictions_list[iG]

            for iG in range(0, num_of_scales):
                # G_list[iG].plot()
                G_list[iG].Utf = G_list[iG].U
                save_GSP_graph(G_list[iG], path.join(filename_for_run, 'GSP_graph_params{:d}.pkl'.format(iG)))
                G_list[iG].R = tf.convert_to_tensor(G_list[iG].R, dtype=model_precision)
                G_list[iG].P = tf.convert_to_tensor(G_list[iG].P, dtype=model_precision)
                G_list[iG].Utf = tf.convert_to_tensor(G_list[iG].Utf, dtype=model_precision)
        else:
            for iG in range(0, num_of_scales):
                G_list.append(load_GSP_graph(path.join(filename_for_run, 'GSP_graph_params{:d}.pkl'.format(iG))))

        return G_list