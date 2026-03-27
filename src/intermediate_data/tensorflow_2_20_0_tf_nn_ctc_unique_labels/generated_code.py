import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.nn.ctc_unique_labels(labels=inputs, name=name)

# Generate random valid input
batch_size = 3
max_label_length = 8
labels_np = np.random.randint(
    low=1,  # Avoid 0 except for padding
    high=10,
    size=(batch_size, max_label_length),
    dtype=np.int32
)

# Add padding zeros to some positions
for i in range(batch_size):
    pad_start = np.random.randint(max_label_length // 2, max_label_length)
    labels_np[i, pad_start:] = 0

labels = tf.constant(labels_np)
example_output = call_func(labels)