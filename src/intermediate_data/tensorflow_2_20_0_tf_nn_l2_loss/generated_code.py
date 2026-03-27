import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.nn.l2_loss(t=inputs, name=name)

# Generate random tensor and call the function
random_tensor = tf.constant(np.random.randn(3, 4).astype(np.float32))
example_output = call_func(inputs=random_tensor)