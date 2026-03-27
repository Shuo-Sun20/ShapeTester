import tensorflow as tf
import numpy as np

def call_func(inputs, axis=None, keepdims=False, name=None):
    return tf.math.reduce_euclidean_norm(
        input_tensor=inputs,
        axis=axis,
        keepdims=keepdims,
        name=name
    )

# Generate random input tensor
np.random.seed(42)
random_input = tf.constant(np.random.randn(3, 4, 5).astype(np.float32))

# Call the function with example parameters
example_output = call_func(
    inputs=random_input,
    axis=[0, 2],
    keepdims=True,
    name="example_reduce_euclidean_norm"
)