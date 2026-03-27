import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    factor,
    value_range=(0, 255),
    data_format=None,
    seed=None
):
    layer = keras.layers.RandomHue(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

# Test input that causes the defect
inputs_array = np.random.rand(8, 32, 32, 3).astype(np.float32)
factor = 0.5
value_range = [0, 1]
data_format = 'channels_first'
seed = None

print("Testing with eager tensor (dynamic shape):")
try:
    dynamic_result = call_func(
        inputs=inputs_array,
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception - {str(e)}")

print("\nTesting with Keras.Input placeholder (static shape):")
try:
    input_placeholder = keras.Input(shape=(32, 32, 3))
    static_result = call_func(
        inputs=input_placeholder,
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: Exception - {str(e)}")