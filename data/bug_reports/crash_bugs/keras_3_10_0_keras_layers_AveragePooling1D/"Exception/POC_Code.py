import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(
    pool_size,
    inputs,
    strides=None,
    padding="valid",
    data_format=None,
    name=None
):
    layer = keras.layers.AveragePooling1D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    output = layer(inputs)
    return output

# Test input parameters
pool_size = 7
strides = 3
padding = 'valid'
data_format = 'channels_first'
name = None

# Create eager tensor input
eager_input = np.random.random((2, 10, 3)).astype(np.float32)
eager_tensor = keras.ops.convert_to_tensor(eager_input)

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(10, 3))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        pool_size=pool_size,
        inputs=eager_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    print(f"Dynamic output shape: {list(eager_output.shape)}")
except Exception as e:
    print(f"Exception with eager tensor: {e}")

print("\nTesting with Keras.Input placeholder:")
try:
    static_output = call_func(
        pool_size=pool_size,
        inputs=placeholder_input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    print(f"Static output shape: {list(static_output.shape)}")
except Exception as e:
    print(f"Exception with placeholder: {e}")