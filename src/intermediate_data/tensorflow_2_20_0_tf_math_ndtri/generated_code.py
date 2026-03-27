import tensorflow as tf

def call_func(inputs, name=None):
    x = inputs[0]
    return tf.math.ndtri(x, name)

# Generate a random tensor with values in the valid range (0,1)
# Avoid values exactly 0 or 1 due to asymptotes at those points
random_tensor = tf.random.uniform(shape=(2, 3), minval=0.01, maxval=0.99, dtype=tf.float32)

# Call the function with the generated tensor
example_output = call_func([random_tensor])