import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.erfinv(x=inputs, name=name)

# Generate valid input tensor for erfinv (values must be in [-1, 1])
input_tensor = tf.random.uniform(shape=(3, 4), minval=-0.99, maxval=0.99, dtype=tf.float32)
example_output = call_func(inputs=input_tensor)