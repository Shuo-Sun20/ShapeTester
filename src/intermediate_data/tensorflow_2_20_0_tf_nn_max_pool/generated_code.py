import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format=None, name=None):
    return tf.nn.max_pool(
        input=inputs,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )

example_input = tf.constant(tf.random.normal(shape=(1, 4, 4, 1), dtype=tf.float32))
example_output = call_func(
    inputs=example_input,
    ksize=2,
    strides=2,
    padding='SAME'
)