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

# Test input parameters
top_cropping = 5
left_cropping = None
bottom_cropping = 6
right_cropping = 7
target_height = None
target_width = 64
data_format = None

# Create test data - eager tensor
eager_input = np.random.rand(4, 32, 32, 3).astype(np.float32)

# Create Keras Input placeholder with same shape
keras_input = keras.Input(shape=(32, 32, 3))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        inputs=eager_input,
        top_cropping=top_cropping,
        left_cropping=left_cropping,
        bottom_cropping=bottom_cropping,
        right_cropping=right_cropping,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

print("\nTesting with Keras Input placeholder:")
try:
    static_output = call_func(
        inputs=keras_input,
        top_cropping=top_cropping,
        left_cropping=left_cropping,
        bottom_cropping=bottom_cropping,
        right_cropping=right_cropping,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")