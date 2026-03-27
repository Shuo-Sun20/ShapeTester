import tensorflow as tf
import numpy as np

def call_func(inputs, k, name=None):
    targets, predictions = inputs
    return tf.math.in_top_k(targets, predictions, k, name)

batch_size = 3
num_classes = 4
k = 2
targets = tf.constant(np.random.randint(0, num_classes, size=batch_size), dtype=tf.int32)
predictions = tf.constant(np.random.randn(batch_size, num_classes).astype(np.float32))
example_output = call_func([targets, predictions], k)