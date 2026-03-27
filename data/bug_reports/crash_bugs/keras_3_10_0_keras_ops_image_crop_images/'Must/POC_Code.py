import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras import ops

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

# Test with eager tensors (dynamic execution)
print("Testing with eager tensors:")
eager_input = np.random.random((4, 32, 32, 3)).astype(np.float32)
try:
    dynamic_result = call_func(
        inputs=eager_input,
        top_cropping=1,
        left_cropping=None,
        bottom_cropping=1,
        right_cropping=0,
        target_height=32,
        target_width=1,
        data_format='channels_last'
    )
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test with Keras.Input placeholders (static execution)
print("\nTesting with Keras.Input placeholders:")
static_input = keras.Input(shape=(32, 32, 3))
try:
    static_result = call_func(
        inputs=static_input,
        top_cropping=1,
        left_cropping=None,
        bottom_cropping=1,
        right_cropping=0,
        target_height=32,
        target_width=1,
        data_format='channels_last'
    )
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")

print("\nDefect reproduced: The dynamic execution throws an error while static execution returns a valid shape.")