import tensorflow as tf
import keras
import numpy as np

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

# Generate random data for example
np.random.seed(42)
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
random_strings = [
    ''.join(np.random.choice(chars, size=3))
    for _ in range(10)
]
data = tf.constant(random_strings)
vocab = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']

example_output = call_func(
    inputs=data,
    vocabulary=vocab,
    num_oov_indices=2,
    output_mode="int"
)