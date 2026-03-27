import tensorflow as tf

def call_func(inputs, variance_epsilon, name=None):
    x, mean, variance, offset, scale = inputs
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name)

# Generate random input tensors for a [batch, depth] layout
batch_size = 4
depth = 6

x = tf.random.normal([batch_size, depth])
mean = tf.random.normal([depth])
variance = tf.abs(tf.random.normal([depth]))  # Ensure variance is non-negative
offset = tf.random.normal([depth])
scale = tf.random.normal([depth])

# Call the function
inputs = [x, mean, variance, offset, scale]
example_output = call_func(inputs, variance_epsilon=0.001)