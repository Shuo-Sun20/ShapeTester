import tensorflow as tf

def call_func(inputs, name=None):
    return tf.linalg.expm(inputs, name=name)

# Create random input tensor
input_tensor = tf.random.normal(shape=(2, 3, 3), dtype=tf.float32)
example_output = call_func(inputs=input_tensor)