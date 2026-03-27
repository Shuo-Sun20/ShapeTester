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

# Test with eager tensor
eager_input = tf.constant(["aaa", "bbb", "ccc", "ddd", "eee"])
eager_output = call_func(
    inputs=eager_input,
    max_tokens=50,
    num_oov_indices=2,
    mask_token=None,
    oov_token="[UNK]",
    vocabulary=["aaa", "bbb", "ccc", "ddd", "eee"],
    idf_weights=None,
    invert=False,
    output_mode="multi_hot",
    pad_to_max_tokens=False,
    sparse=False,
    encoding="utf-8",
    name=None
)

# Test with Keras.Input placeholder
placeholder_input = keras.Input(shape=(5,), dtype=tf.string)
placeholder_output = call_func(
    inputs=placeholder_input,
    max_tokens=50,
    num_oov_indices=2,
    mask_token=None,
    oov_token="[UNK]",
    vocabulary=["aaa", "bbb", "ccc", "ddd", "eee"],
    idf_weights=None,
    invert=False,
    output_mode="multi_hot",
    pad_to_max_tokens=False,
    sparse=False,
    encoding="utf-8",
    name=None
)

print(f"Eager tensor input shape: {eager_input.shape}")
print(f"Dynamic output shape (eager): {eager_output.shape}")
print(f"Placeholder input shape: {placeholder_input.shape}")
print(f"Static output shape (placeholder): {placeholder_output.shape}")

# Demonstrate the inconsistency
print(f"\nInconsistency detected:")
print(f"Dynamic output shape: {list(eager_output.shape)}")
print(f"Static output shape: {list(placeholder_output.shape)}")