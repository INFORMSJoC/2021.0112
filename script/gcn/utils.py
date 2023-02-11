import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def sparse_tensor_dense_tensordot(sp_a, b, axes, name=None):
    import tensorflow as tf
    r"""Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
        a: `SparseTensor` of type `float32` or `float64`.
        b: `Tensor` with the same type as `a`.
        axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
         If axes is a scalar, sum over the last N axes of a and the first N axes
         of b in order.
         If axes is a list or `Tensor` the first and second row contain the set of
         unique integers specifying axes along which the contraction is computed,
         for `a` and `b`, respectively. The number of axes for `a` and `b` must
         be equal.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` with the same type as `a`.
    Raises:
        ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
        IndexError: If the values in axes exceed the rank of the corresponding
            tensor.
    """

    def _tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            # print(sess.run(rank_a), sess.run(axes))
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.name_scope(name, "SparseTensorDenseTensordot", [sp_a, b, axes]) as name:
#         a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
        sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(sp_a, sp_a_axes)
        b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
                b, b_axes, True)
        ab_matmul = tf.sparse_tensor_dense_matmul(sp_a_reshape, b_reshape)
        if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(ab_matmul, sp_a_free_dims + b_free_dims, name=name)
        else:
            sp_a_free_dims = tf.convert_to_tensor(sp_a_free_dims, dtype=tf.int32)
            b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.int32)
            product = tf.reshape(
                    ab_matmul, tf.concat([sp_a_free_dims, b_free_dims], 0), name=name)
            if sp_a_free_dims_static is not None and b_free_dims_static is not None:
                product.set_shape(sp_a_free_dims_static + b_free_dims_static)
            return product