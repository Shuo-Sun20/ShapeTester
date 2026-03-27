import tensorflow as tf

def call_func(inputs, name="collapse_repeated_labels"):
    labels, seq_length = inputs
    collapsed_labels, new_seq_length = tf.nn.collapse_repeated(labels, seq_length, name)
    return [collapsed_labels, new_seq_length]

# Generate random input data
batch_size = 3
max_seq_len = 7
labels = tf.random.uniform(shape=[batch_size, max_seq_len], minval=0, maxval=4, dtype=tf.int32)
seq_length = tf.random.uniform(shape=[batch_size], minval=1, maxval=max_seq_len+1, dtype=tf.int32)

# Call the function
example_output = call_func([labels, seq_length])