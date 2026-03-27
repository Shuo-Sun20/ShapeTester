import tensorflow as tf

def call_func(inputs, sp_weights=None, combiner='mean', max_norm=None, name=None, allow_fast_lookup=False):
    """
    Wraps tf.nn.embedding_lookup_sparse.
    
    Args:
        inputs: List containing [params, sp_ids] where params is the embedding tensor(s)
                and sp_ids is a 2D SparseTensor/RaggedTensor of indices.
        sp_weights: Optional SparseTensor/RaggedTensor of weights.
        combiner: Reduction method ('sum', 'mean', or 'sqrtn').
        max_norm: If not None, each embedding is clipped if its l2-norm is larger than this value.
        name: Optional name for the op.
        allow_fast_lookup: Optional boolean for enabling fast lookup.
    
    Returns:
        A dense tensor of combined embeddings.
    """
    params, sp_ids = inputs[0], inputs[1]
    return tf.nn.embedding_lookup_sparse(
        params=params,
        sp_ids=sp_ids,
        sp_weights=sp_weights,
        combiner=combiner,
        max_norm=max_norm,
        name=name,
        allow_fast_lookup=allow_fast_lookup
    )

# Generate random inputs
tf.random.set_seed(42)
params = tf.random.uniform(shape=(10, 8), minval=-1.0, maxval=1.0, dtype=tf.float32)

# Create a sparse tensor with int64 indices
indices = tf.constant([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1]], dtype=tf.int64)
values = tf.constant([0, 1, 3, 2, 4], dtype=tf.int64)
dense_shape = tf.constant([3, 3], dtype=tf.int64)
sp_ids = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

# Call function and store output
example_output = call_func(inputs=[params, sp_ids], combiner='sum')