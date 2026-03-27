import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format='NHWC', name=None):
    return tf.nn.max_pool2d(
        input=inputs,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )

input_tensor = tf.random.normal(shape=(2, 8, 8, 3))
example_output = call_func(
    inputs=input_tensor,
    ksize=(2, 2),
    strides=(2, 2),
    padding='VALID',
    data_format='NHWC',
    name=None
)