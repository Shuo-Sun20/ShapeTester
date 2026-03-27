import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

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

# Test parameters
test_params = {
    'max_tokens': None,
    'num_oov_indices': 0,
    'mask_token': -1,
    'oov_token': -999,
    'vocabulary': list(range(100)),
    'vocabulary_dtype': 'int64',
    'idf_weights': None,
    'invert': False,
    'output_mode': 'int',
    'sparse': False,
    'pad_to_max_tokens': False,
    'name': None
}

# Test input data containing OOV values
test_input = np.array([[12, 1138, 42], [42, 1000, 36]])

print("Testing with eager tensors:")
try:
    dynamic_output = call_func(**test_params, inputs=test_input)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling IntegerLookup.call().\n\n{e}")

print("\nTesting with Keras.Input placeholders:")
# Create Keras Input with same shape as test_input
input_placeholder = keras.Input(shape=(3,), dtype='int32')
static_output = call_func(**test_params, inputs=input_placeholder)
print(f"Static output shape: {static_output.shape}")

print(f"\nTest input shape: {test_input.shape}")
print(f"Test input data:\n{test_input}")