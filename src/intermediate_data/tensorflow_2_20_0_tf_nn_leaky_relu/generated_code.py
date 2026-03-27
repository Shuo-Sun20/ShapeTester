import tensorflow as tf
import numpy as np

def call_func(inputs, alpha=0.2, name=None):
    return tf.nn.leaky_relu(features=inputs, alpha=alpha, name=name)

# Generate random tensor as input
random_tensor = tf.constant(np.random.randn(3, 4, 5).astype(np.float32))
example_output = call_func(inputs=random_tensor, alpha=0.3)