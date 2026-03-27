import tensorflow as tf
import numpy as np

def call_func(inputs, ksize, strides, padding="VALID", data_format="NHWC", name=None):
    input_tensor = inputs[0] if isinstance(inputs, list) else inputs
    return tf.nn.avg_pool(
        input=input_tensor,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )

# Generate random input tensor
np.random.seed(42)
batch_size, height, width, channels = 2, 4, 4, 3
input_shape = [batch_size, height, width, channels]
random_tensor = np.random.rand(*input_shape).astype(np.float32)

# Call function and save output
example_output = call_func(
    inputs=[random_tensor],
    ksize=[1, 2, 2, 1],
    strides=[1, 1, 1, 1],
    padding="SAME",
    data_format="NHWC"
)