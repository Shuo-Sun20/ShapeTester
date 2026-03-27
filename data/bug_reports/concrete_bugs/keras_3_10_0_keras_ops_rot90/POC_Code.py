import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, k=1, axes=(0, 1)):
    return keras.ops.rot90(array=inputs, k=k, axes=axes)

# Test input parameters
input_shape = (3, 4, 5)
k = -3
axes = [0, 1]

# Create test data
test_array = np.random.random(input_shape)

# Test with eager tensor (dynamic shape)
eager_tensor = keras.ops.convert_to_tensor(test_array)
dynamic_result = call_func(eager_tensor, k=k, axes=axes)
dynamic_shape = dynamic_result.shape

# Test with Keras.Input placeholder (static shape)
input_placeholder = keras.Input(shape=input_shape)
static_result = call_func(input_placeholder, k=k, axes=axes)
static_shape = static_result.shape

print(f"Input shape: {input_shape}")
print(f"k: {k}, axes: {axes}")
print(f"Dynamic output shape: {dynamic_shape}")
print(f"Static output shape: {static_shape}")
print(f"Shapes match: {dynamic_shape == static_shape}")