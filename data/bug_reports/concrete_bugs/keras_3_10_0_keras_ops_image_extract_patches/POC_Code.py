import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Input
from keras.models import Model

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

# Test input parameters
input_shape = (2, 20, 20, 3)
size = [3, 5]
strides = None
dilation_rate = [1, 1]
padding = "valid"
data_format = "channels_first"

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
eager_input = np.random.random(input_shape).astype("float32")
dynamic_output = call_func(
    inputs=eager_input,
    size=size,
    strides=strides,
    dilation_rate=dilation_rate,
    padding=padding,
    data_format=data_format
)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
input_placeholder = Input(shape=input_shape[1:])  # Remove batch dimension for Input
model_input = input_placeholder

# Create a model to get static shape information
def extract_patches_layer(x):
    return call_func(
        inputs=x,
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format
    )

output = extract_patches_layer(model_input)
model = Model(inputs=model_input, outputs=output)

# Get static output shape
static_output_shape = model.output_shape
print(f"Static output shape: {static_output_shape}")

# Show the inconsistency
print(f"\nInconsistency detected:")
print(f"Dynamic shape: {dynamic_output.shape}")
print(f"Static shape: {static_output_shape}")