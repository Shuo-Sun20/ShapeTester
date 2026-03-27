import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf

def call_func(bin_boundaries=None, num_bins=None, epsilon=0.01, output_mode="int", sparse=False, dtype=None, name=None, inputs=None):
    layer = keras.layers.Discretization(
        bin_boundaries=bin_boundaries,
        num_bins=num_bins,
        epsilon=epsilon,
        output_mode=output_mode,
        sparse=sparse,
        dtype=dtype,
        name=name
    )
    return layer(inputs)

# Test input parameters based on the defect description
bin_boundaries = [-2.0, -1.0]
num_bins = None
epsilon = 0.01
output_mode = "one_hot"
sparse = False
dtype = None
name = None

# Create test input data with shape (5, 4)
input_data = np.random.uniform(-3.0, 2.0, size=(5, 4))

# Test with eager tensors (dynamic shape)
eager_tensor = tf.constant(input_data, dtype=tf.float32)
dynamic_result = call_func(
    bin_boundaries=bin_boundaries,
    num_bins=num_bins,
    epsilon=epsilon,
    output_mode=output_mode,
    sparse=sparse,
    dtype=dtype,
    name=name,
    inputs=eager_tensor
)

print("Dynamic output shape:", dynamic_result.shape.as_list())

# Test with Keras.Input placeholders (static shape)
placeholder_input = keras.Input(shape=(4,), dtype=tf.float32)
static_result = call_func(
    bin_boundaries=bin_boundaries,
    num_bins=num_bins,
    epsilon=epsilon,
    output_mode=output_mode,
    sparse=sparse,
    dtype=dtype,
    name=name,
    inputs=placeholder_input
)

print("Static output shape:", static_result.shape)

# Demonstrate the defect
print(f"Dynamic shape: {dynamic_result.shape.as_list()}")
print(f"Static shape: {static_result.shape}")
print(f"Shapes are inconsistent: {dynamic_result.shape.as_list() != static_result.shape.as_list()}")