import numpy as np
import keras

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

vocab = [12, 36, 1138, 42]
data = np.random.randint(0, 2000, size=(3, 5))
example_output = call_func(vocabulary=vocab, inputs=data)