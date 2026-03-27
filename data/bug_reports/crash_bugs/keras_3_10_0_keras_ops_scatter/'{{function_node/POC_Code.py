import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, shape):
    indices, values = inputs
    return keras.ops.scatter(indices, values, shape)

# Test input that causes the defect
indices = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [2, 2]])  # shape (5, 2)
values = np.array([1., 2., 3., 4., 5.])  # shape (5,)
shape = [1, 1]

print("Testing with eager tensors:")
try:
    # Call with eager tensors
    result_eager = call_func([indices, values], shape)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Dynamic output shape error: {e}")

print("\nTesting with Keras.Input placeholders:")
# Call with Keras.Input placeholders
indices_input = keras.Input(shape=(2,), name='indices')
values_input = keras.Input(shape=(), name='values')

# Create a functional model to get static shape
outputs = call_func([indices_input, values_input], shape)
model = keras.Model(inputs=[indices_input, values_input], outputs=outputs)

print(f"Static output shape: {model.output.shape}")