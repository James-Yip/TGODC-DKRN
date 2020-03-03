import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import itertools
import math


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def pt_default_weight_init(shape, name=None):
    stdv = 1. / math.sqrt(shape[1])
    initial = tf.random_uniform(shape, minval=-stdv, maxval=stdv, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def pt_default_bias_init(shape, name=None):
    stdv = 1. / math.sqrt(shape[0])
    initial = tf.random_uniform(shape, minval=-stdv, maxval=stdv, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

# def dot(x, y, x_is_sparse=False, y_is_sparse=False):
#     """Wrapper for tf.matmul (sparse vs dense)."""
#     res = tf.matmul(x, y,a_is_sparse=x_is_sparse, b_is_sparse=y_is_sparse)
#     return res


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def batch_dot(x, y, sparse=False, only_first=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        # res = tf.sparse_tensor_dense_matmul(x, y)
        if only_first:
            res = tf.scan(lambda a, b: tf.sparse_tensor_dense_matmul(x,b), y)
        else:
            res = tf.scan(lambda a, b: tf.matmul(x,b, a_is_sparse=True, b_is_sparse=True), y)
    else:
        # res = tf.matmul(x, y)
        res = tf.scan(lambda a, b: tf.matmul(b,y), x)
    return res


# useless for tgodc
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_sigmoid_cross_entropy(preds, labels, mask):
    """sigmoid cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

# useless for tgodc
def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def sparse_matrix(matrix):
    return sp.coo_matrix(matrix)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adjacency(adjacency):
    """Symmetrically normalize adjacency matrix."""
    adjacency = sp.coo_matrix(adjacency)
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_asymmetric_adjacency(adjacency):
    """Symmetrically normalize adjacency matrix."""
    adjacency = sp.coo_matrix(adjacency)
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).tocoo()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adjacency(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj1(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    ones_matrix = np.ones(shape=[adj.shape[0], adj.shape[0]], dtype=np.float32)
    diagonal = sp.eye(adj.shape[0])
    adj = adj * (ones_matrix - diagonal)
    adj_normalized = normalize_adjacency(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_asymmetric_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_asymmetric_adjacency(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_asymmetric_adj1(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    ones_matrix = np.ones(shape=[adj.shape[0], adj.shape[0]], dtype=np.float32)
    diagonal = sp.eye(adj.shape[0])
    adj = adj * (ones_matrix - diagonal)
    adj_normalized = normalize_asymmetric_adjacency(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def generate_spare_diagonal_matrix(shape):
    # Sparse matrix with ones on diagonal
    return sp.eye(shape)


def generate_dense_diagonal_matrix(shape):
    return sp.eye(shape).todense()

def chebyshev_polynomials(adj, max_degree):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adjacency(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, max_degree+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adjacency_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def preprocess_adjacency_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


# form book learning graph neural network
# def adjacency_normalization(adjacency):
#     """计算 L=D^-0.5 * (A+I) * D^-0.5"""
#     adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
#     degree = np.array(adjacency.sum(1))
#     d_hat = sp.diags(np.power(degree, -0.5).flatten())
#     return d_hat.dot(adjacency).dot(d_hat).tocoo()


# def build_adjacency(adjacency_dict, num_nodes):
#     """根据邻接表创建邻接矩阵"""
#     edge_index = []
#     # num_nodes = len(adjacency_dict) # 应该由外面传入点的数量
#     for src, dst in adjacency_dict.items():
#         edge_index.extend([src, v] for v in dst)
#         edge_index.extend([v, src] for v in dst)
#     # 去除重复的边
#     edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
#     edge_index = np.asarray(edge_index)
#     adjacency = sp.coo_matrix((np.ones(len(edge_index)),
#                                (edge_index[:, 0], edge_index[:, 1])),
#                               shape=(num_nodes, num_nodes), dtype="float32")
#     return adjacency