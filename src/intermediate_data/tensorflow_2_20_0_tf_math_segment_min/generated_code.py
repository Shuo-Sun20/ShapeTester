import tensorflow as tf
import numpy as np

def call_func(inputs, segment_ids, name=None):
    data = inputs
    output = tf.math.segment_min(data=data, segment_ids=segment_ids, name=name)
    return output

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Generate random data tensor
data_tensor = tf.constant(np.random.randint(0, 20, size=(6, 3)), dtype=tf.int32)
# Generate sorted segment_ids (as required on CPU)
segment_ids = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)

# Call function with valid inputs
example_output = call_func(inputs=data_tensor, segment_ids=segment_ids)