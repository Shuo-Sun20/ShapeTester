import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, num_classes, axis=-1, dtype=None, sparse=False):
    # Direct API call since keras.ops.multi_hot is a function
    return keras.ops.multi_hot(
        inputs=inputs,
        num_classes=num_classes,
        axis=axis,
        dtype=dtype,
        sparse=sparse
    )

# Test input that causes the defect
# Create eager tensor with shape (10,)
eager_inputs = keras.ops.convert_to_tensor(np.random.randint(0, 1, size=(10,)))
num_classes = 1
axis = 1
dtype = None
sparse = False

print("Testing with eager tensor:")
print(f"Eager tensor shape: {eager_inputs.shape}")

try:
    # Call with eager tensor
    dynamic_output = call_func(eager_inputs, num_classes, axis, dtype, sparse)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with Keras.Input placeholder:")
# Create Keras.Input placeholder with same shape
placeholder_inputs = keras.Input(shape=(10,))
print(f"Placeholder shape: {placeholder_inputs.shape}")

try:
    # Call with placeholder
    static_output = call_func(placeholder_inputs, num_classes, axis, dtype, sparse)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output error: {e}")