import tensorflow as tf
import numpy as np

def call_func(inputs, ids, max_norm=None, name=None):
    params = inputs
    result = tf.nn.embedding_lookup(params, ids, max_norm=max_norm, name=name)
    return result

# Generate random inputs
np.random.seed(42)
params_tensor = np.random.randn(5, 3).astype(np.float32)
ids_tensor = np.array([0, 2, 4], dtype=np.int32)

# Call the function
example_output = call_func(params_tensor, ids_tensor)