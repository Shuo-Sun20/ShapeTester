import tensorflow as tf
import numpy as np

def call_func(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values=None, remove_accidental_hits=False, name=None):
    return tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        name=name
    )

# Generate random inputs
batch_size = 32
dim = 128
num_classes = 10000
num_true = 2
num_sampled = 50

weights = tf.random.normal([num_classes, dim])
biases = tf.random.normal([num_classes])
labels = tf.random.uniform([batch_size, num_true], 0, num_classes, dtype=tf.int64)
inputs = tf.random.normal([batch_size, dim])

# Call function
example_output = call_func(
    weights=weights,
    biases=biases,
    labels=labels,
    inputs=inputs,
    num_sampled=num_sampled,
    num_classes=num_classes,
    num_true=num_true,
    sampled_values=None,
    remove_accidental_hits=False
)