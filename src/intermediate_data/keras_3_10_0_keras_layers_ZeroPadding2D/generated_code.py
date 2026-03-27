import numpy as np
import keras

def call_func(padding, data_format, inputs):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    layer = keras.layers.ZeroPadding2D(
        padding=padding,
        data_format=data_format
    )
    
    output_tensor = layer(input_tensor)
    return output_tensor

input_tensor = np.random.randn(2, 32, 32, 3).astype(np.float32)
example_output = call_func(
    padding=((2, 2), (1, 1)),
    data_format="channels_last",
    inputs=input_tensor
)