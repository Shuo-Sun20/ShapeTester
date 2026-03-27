import tensorflow as tf
import numpy as np

def call_func(inputs, num_classes=None, weights=None, dtype=None, name=None):
    labels = inputs[0]
    predictions = inputs[1]
    return tf.math.confusion_matrix(labels, predictions, num_classes=num_classes, weights=weights, dtype=dtype, name=name)

labels = tf.constant([1, 2, 0, 1, 2, 0], dtype=tf.int32)
predictions = tf.constant([2, 2, 0, 1, 1, 0], dtype=tf.int32)
example_output = call_func([labels, predictions], num_classes=3)