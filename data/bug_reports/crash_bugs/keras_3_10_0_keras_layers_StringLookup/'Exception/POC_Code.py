import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(
    inputs,
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token="[UNK]",
    vocabulary=None,
    idf_weights=None,
    invert=False,
    output_mode="int",
    pad_to_max_tokens=False,
    sparse=False,
    encoding="utf-8",
    name=None
):
    layer = keras.layers.StringLookup(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        idf_weights=idf_weights,
        invert=invert,
        output_mode=output_mode,
        pad_to_max_tokens=pad_to_max_tokens,
        sparse=sparse,
        encoding=encoding,
        name=name
    )
    output = layer(inputs)
    return output

# Test parameters that cause the defect
test_params = {
    'max_tokens': 20,
    'num_oov_indices': 0,
    'mask_token': None,
    'oov_token': '[UNK]',
    'vocabulary': ['a', 'b', 'c'],
    'idf_weights': None,
    'invert': False,
    'output_mode': 'int',
    'pad_to_max_tokens': False,
    'sparse': False,
    'encoding': 'utf-8',
    'name': None
}

# Create eager tensor input with OOV values
eager_input = tf.constant(['aaa', 'bbb', 'ccc', 'ddd', 'eee'], dtype=tf.string)

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(5,), dtype=tf.string)

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_input, **test_params)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception - {str(e)}")

print("\nTesting with Keras.Input placeholder:")
try:
    static_output = call_func(placeholder_input, **test_params)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception - {str(e)}")