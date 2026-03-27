import tensorflow as tf

def call_func(inputs, axes, frequency_weights=None, keepdims=False, name=None):
    x = inputs
    weighted_mean, weighted_variance = tf.nn.weighted_moments(
        x=x,
        axes=axes,
        frequency_weights=frequency_weights,
        keepdims=keepdims,
        name=name
    )
    return [weighted_mean, weighted_variance]

# Generate random input tensors
x = tf.random.normal(shape=[2, 3, 4], dtype=tf.float32)
frequency_weights = tf.abs(tf.random.normal(shape=[2, 3, 4], dtype=tf.float32)) + 0.1
axes = [0, 1]
keepdims = False

# Call the function
example_output = call_func(inputs=x, axes=axes, frequency_weights=frequency_weights, keepdims=keepdims)