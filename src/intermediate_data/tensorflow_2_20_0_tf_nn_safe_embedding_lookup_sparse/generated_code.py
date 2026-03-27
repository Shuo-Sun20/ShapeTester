import tensorflow as tf
import numpy as np

def call_func(inputs, combiner="mean", default_id=None, max_norm=None, name=None, allow_fast_lookup=False):
    if len(inputs) == 3:
        embedding_weights, sparse_ids, sparse_weights = inputs
    elif len(inputs) == 2:
        embedding_weights, sparse_ids = inputs
        sparse_weights = None
    else:
        raise ValueError("inputs must contain 2 or 3 elements")
    
    return tf.nn.safe_embedding_lookup_sparse(
        embedding_weights=embedding_weights,
        sparse_ids=sparse_ids,
        sparse_weights=sparse_weights,
        combiner=combiner,
        default_id=default_id,
        max_norm=max_norm,
        name=name,
        allow_fast_lookup=allow_fast_lookup
    )

# Generate random embedding weights (vocabulary size 100, embedding dim 16)
embedding_weights = tf.random.uniform(shape=[100, 16], minval=-1.0, maxval=1.0)

# Create sparse_ids with shape [3, 4] (batch_size=3, max_features=4)
indices = [[0, 0], [0, 1], [1, 0], [2, 3], [2, 1]]
values = [5, -1, 30, 15, 7]  # Includes invalid ID -1
dense_shape = [3, 4]
sparse_ids = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

# Create sparse_weights with same sparsity pattern
weight_values = [1.5, 0.0, 0.8, 2.0, 1.2]  # Includes weight 0.0 (will be ignored)
sparse_weights = tf.SparseTensor(indices=indices, values=weight_values, dense_shape=dense_shape)

example_output = call_func(
    inputs=[embedding_weights, sparse_ids, sparse_weights],
    combiner="mean",
    default_id=0,
    max_norm=1.0,
    allow_fast_lookup=True
)