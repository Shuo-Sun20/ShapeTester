import tensorflow as tf
import numpy as np

def call_func(inputs, pos_weight, name=None):
    labels = inputs[0]
    logits = inputs[1]
    return tf.nn.weighted_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        pos_weight=pos_weight,
        name=name
    )

# Generate random valid inputs
np.random.seed(42)
batch_size = 4
logits = tf.constant(np.random.randn(batch_size).astype(np.float32))
labels = tf.constant(np.random.uniform(0, 1, batch_size).astype(np.float32))
pos_weight = tf.constant(1.5, dtype=tf.float32)

example_output = call_func(
    inputs=[labels, logits],
    pos_weight=pos_weight
)