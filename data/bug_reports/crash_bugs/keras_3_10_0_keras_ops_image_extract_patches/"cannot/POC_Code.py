import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras import ops

def call_func(
    inputs,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None
):
    return keras.ops.image.extract_patches(
        images=inputs,
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format
    )

# Test input that causes the defect
test_input = np.random.random((2, 20, 20, 3)).astype('float32')
size = [1, 1]
strides = [1, 1]
dilation_rate = 1
padding = 'valid'
data_format = None

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
dynamic_result = call_func(
    inputs=test_input,
    size=size,
    strides=strides,
    dilation_rate=dilation_rate,
    padding=padding,
    data_format=data_format
)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
try:
    input_placeholder = keras.Input(shape=(20, 20, 3))
    static_result = call_func(
        inputs=input_placeholder,
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format
    )
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")

print(f"\nExpected behavior: Both should produce the same shape pattern")
print(f"Actual behavior: Dynamic and static shapes are inconsistent")