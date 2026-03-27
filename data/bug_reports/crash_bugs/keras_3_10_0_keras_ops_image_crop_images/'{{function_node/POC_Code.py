import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    top_cropping=None,
    left_cropping=None,
    bottom_cropping=None,
    right_cropping=None,
    target_height=None,
    target_width=None,
    data_format=None
):
    return keras.ops.image.crop_images(
        images=inputs,
        top_cropping=top_cropping,
        left_cropping=left_cropping,
        bottom_cropping=bottom_cropping,
        right_cropping=right_cropping,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )

# Test with eager tensors
eager_input = np.random.rand(4, 32, 32, 3).astype(np.float32)
print("Input shape:", eager_input.shape)

try:
    # This should fail with the dynamic shape error
    dynamic_output = call_func(
        inputs=eager_input,
        top_cropping=7,
        left_cropping=5,
        bottom_cropping=None,
        right_cropping=None,
        target_height=64,
        target_width=24,
        data_format=None
    )
    print("Dynamic output shape:", dynamic_output.shape)
except Exception as e:
    print("Dynamic execution error:", str(e))

# Test with Keras.Input placeholders (static shape)
placeholder_input = keras.Input(shape=(32, 32, 3))
print("Placeholder input shape:", placeholder_input.shape)

try:
    # This should work and give the expected static shape
    static_output = call_func(
        inputs=placeholder_input,
        top_cropping=7,
        left_cropping=5,
        bottom_cropping=None,
        right_cropping=None,
        target_height=64,
        target_width=24,
        data_format=None
    )
    print("Static output shape:", static_output.shape)
except Exception as e:
    print("Static execution error:", str(e))

print("\nDefect reproduced: Dynamic and static shapes are inconsistent")
print("Expected behavior: Both should have shape [None, 64, 24, 3] or both should fail")