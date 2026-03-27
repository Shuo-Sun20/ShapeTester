import tensorflow as tf
from keras.layers import Lambda

def call_func(function, output_shape=None, mask=None, arguments=None, inputs=None):
    layer_instance = Lambda(function=function, output_shape=output_shape, mask=mask, arguments=arguments)
    return layer_instance(inputs)

input_tensor = tf.random.normal(shape=(2, 3))
square_func = lambda x: x ** 2
example_output = call_func(function=square_func, inputs=input_tensor)