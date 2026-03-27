import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.imag(input=inputs, name=name)

input_tensor = tf.complex(
    tf.random.normal(shape=(3, 3), dtype=tf.float32),
    tf.random.normal(shape=(3, 3), dtype=tf.float32)
)
example_output = call_func(inputs=input_tensor)