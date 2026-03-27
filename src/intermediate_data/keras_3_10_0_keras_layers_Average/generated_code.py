import keras
import numpy as np

def call_func(constructor_kwargs, inputs):
    layer_instance = keras.layers.Average(**constructor_kwargs)
    return layer_instance(inputs)

input_shape = (2, 3, 4)
tensor_list = [
    np.random.rand(*input_shape).astype('float32'),
    np.random.rand(*input_shape).astype('float32')
]
example_output = call_func({}, tensor_list)