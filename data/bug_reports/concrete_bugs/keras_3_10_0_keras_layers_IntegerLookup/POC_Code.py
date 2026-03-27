import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf

def call_func(
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token=-1,
    vocabulary=None,
    vocabulary_dtype="int64",
    idf_weights=None,
    invert=False,
    output_mode="int",
    sparse=False,
    pad_to_max_tokens=False,
    name=None,
    inputs=None
):
    layer = keras.layers.IntegerLookup(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        vocabulary_dtype=vocabulary_dtype,
        idf_weights=idf_weights,
        invert=invert,
        output_mode=output_mode,
        sparse=sparse,
        pad_to_max_tokens=pad_to_max_tokens,
        name=name
    )
    return layer(inputs)

# Test input parameters
test_params = {
    'max_tokens': None,
    'num_oov_indices': 3,
    'mask_token': -1,
    'oov_token': -999,
    'vocabulary': [12, 36, 1138, 42],
    'vocabulary_dtype': 'int64',
    'idf_weights': None,
    'invert': False,
    'output_mode': 'one_hot',
    'sparse': False,
    'pad_to_max_tokens': False,
    'name': None
}

# Create test input data (2, 3) shape
input_data = np.array([[12, 36, 1138], [42, 999, 12]], dtype=np.int64)

# Test with eager tensors
eager_tensor = tf.constant(input_data)
dynamic_output = call_func(**test_params, inputs=eager_tensor)
print("Dynamic output shape (eager tensor):", dynamic_output.shape.as_list())

# Test with Keras.Input placeholders
placeholder_input = keras.Input(shape=(3,), dtype='int64')
static_output = call_func(**test_params, inputs=placeholder_input)
print("Static output shape (placeholder):", static_output.shape)

print("\nShape mismatch detected:")
print(f"Dynamic shape: {dynamic_output.shape.as_list()}")
print(f"Static shape: {static_output.shape}")