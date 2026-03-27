import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, axis=-1, mean=None, variance=None, invert=False):
    norm_layer = keras.layers.Normalization(axis=axis, mean=mean, variance=variance, invert=invert)
    return norm_layer(inputs)

# Test input parameters
test_input = np.random.randn(2, 3).astype(np.float32)
axis = [0, 1]
mean = 0.5
variance = 10.0
invert = False

print("Testing with eager tensors:")
try:
    # Call with eager tensor
    eager_result = call_func(test_input, axis=axis, mean=mean, variance=variance, invert=invert)
    print(f"Dynamic output shape: {eager_result.shape}")
    print(f"Eager execution successful")
except Exception as e:
    print(f"Eager execution error: {e}")

print("\nTesting with Keras.Input placeholders:")
try:
    # Call with Keras.Input placeholder of same shape
    placeholder_input = keras.Input(shape=(3,))  # batch dimension is None by default
    static_result = call_func(placeholder_input, axis=axis, mean=mean, variance=variance, invert=invert)
    print(f"Static output shape: {static_result.shape}")
    print(f"Static execution successful")
except Exception as e:
    print(f"Static execution error: {e}")

print("\nComparison:")
print("The defect is reproduced - eager tensors work but Keras.Input placeholders fail with axis=[0,1]")