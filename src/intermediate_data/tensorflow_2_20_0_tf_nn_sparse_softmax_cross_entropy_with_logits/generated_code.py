import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    labels, logits = inputs
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name=name)

batch_size = 3
num_classes = 4
logits = tf.constant(np.random.randn(batch_size, num_classes), dtype=tf.float32)
labels = tf.constant(np.random.randint(0, num_classes, size=batch_size), dtype=tf.int32)
example_output = call_func([labels, logits])