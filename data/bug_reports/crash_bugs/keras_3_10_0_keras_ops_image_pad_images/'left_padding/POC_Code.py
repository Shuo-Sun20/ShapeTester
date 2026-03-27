import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    top_padding=None,
    left_padding=None,
    bottom_padding=None,
    right_padding=None,
    target_height=None,
    target_width=None,
    data_format=None
):
    return keras.ops.image.pad_images(
        images=inputs[0],
        top_padding=top_padding,
        left_padding=left_padding,
        bottom_padding=bottom_padding,
        right_padding=right_padding,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )

# Test input that causes the defect
test_inputs = [np.random.random((2, 15, 25, 3))]
top_padding = 1
left_padding = None
bottom_padding = 2
right_padding = 5
target_height = None
target_width = 25
data_format = 'channels_last'

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
try:
    dynamic_result = call_func(
        inputs=test_inputs,
        top_padding=top_padding,
        left_padding=left_padding,
        bottom_padding=bottom_padding,
        right_padding=right_padding,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
try:
    input_placeholder = keras.Input(shape=(15, 25, 3))
    static_result = call_func(
        inputs=[input_placeholder],
        top_padding=top_padding,
        left_padding=left_padding,
        bottom_padding=bottom_padding,
        right_padding=right_padding,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")