import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
from keras.layers import Lambda
from keras import Input

def call_func(function, output_shape=None, mask=None, arguments=None, inputs=None):
    layer_instance = Lambda(function=function, output_shape=output_shape, mask=mask, arguments=arguments)
    return layer_instance(inputs)

# Define the identity function
def identity(x):
    return x

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
dynamic_result = call_func(
    function=identity,
    output_shape=[None, 4],
    mask=None,
    arguments=None,
    inputs=eager_tensor
)

# Test with Keras Input placeholder
input_placeholder = Input(shape=(3,))  # same shape as eager tensor
static_result = call_func(
    function=identity,
    output_shape=[None, 4],
    mask=None,
    arguments=None,
    inputs=input_placeholder
)

print("Dynamic output shape:", dynamic_result.shape.as_list())
print("Static output shape:", static_result.shape.as_list())