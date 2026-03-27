import keras
import numpy as np

def call_func(inputs, size=(2, 2, 2), data_format=None):
    layer_instance = keras.layers.UpSampling3D(size=size, data_format=data_format)
    output = layer_instance(inputs)
    return output

input_tensor = np.random.rand(2, 1, 2, 1, 3).astype('float32')
example_output = call_func(input_tensor, size=(2, 2, 2), data_format="channels_last")