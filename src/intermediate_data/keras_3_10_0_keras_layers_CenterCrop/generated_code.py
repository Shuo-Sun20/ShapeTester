import keras
import numpy as np

def call_func(height, width, inputs, data_format=None, name=None):
    layer_instance = keras.layers.CenterCrop(height=height, width=width, data_format=data_format, name=name)
    output_tensor = layer_instance(inputs)
    return output_tensor

input_tensor = np.random.randn(2, 256, 256, 3).astype(np.float32)
example_output = call_func(height=224, width=224, inputs=input_tensor, data_format='channels_last')