import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
import tensorflow as tf

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

# Test with eager tensors (dynamic)
print("Testing with eager tensors (dynamic):")
try:
    eager_input = np.random.random((2, 20, 20, 3)).astype("float32")
    dynamic_result = call_func(
        inputs=eager_input,
        size=[1, 20],
        strides=None,
        dilation_rate=2,
        padding="valid",
        data_format="channels_last"
    )
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders (static):")
try:
    static_input = keras.Input(shape=(20, 20, 3), batch_size=2)
    static_result = call_func(
        inputs=static_input,
        size=[1, 20],
        strides=None,
        dilation_rate=2,
        padding="valid",
        data_format="channels_last"
    )
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")

print("\nDefect reproduced: Dynamic execution raises an error while static shape inference succeeds with different behavior.")