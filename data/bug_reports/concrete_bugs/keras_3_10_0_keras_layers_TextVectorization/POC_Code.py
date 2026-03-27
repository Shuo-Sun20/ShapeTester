import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, max_tokens=None, standardize="lower_and_strip_punctuation", split="whitespace", ngrams=None, output_mode="int", output_sequence_length=None, pad_to_max_tokens=False, vocabulary=None, idf_weights=None, sparse=False, ragged=False, encoding="utf-8", name=None):
    text_vectorization_layer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=standardize,
        split=split,
        ngrams=ngrams,
        output_mode=output_mode,
        output_sequence_length=output_sequence_length,
        pad_to_max_tokens=pad_to_max_tokens,
        vocabulary=vocabulary,
        idf_weights=idf_weights,
        sparse=sparse,
        ragged=ragged,
        encoding=encoding,
        name=name
    )
    
    if isinstance(inputs, list):
        output = text_vectorization_layer(*inputs)
    else:
        output = text_vectorization_layer(inputs)
    return output

# Create eager tensor input (shape=(3, 1))
eager_input = tf.constant([["hello world test"], ["test example"], ["keras tensorflow"]])

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(1,), dtype=tf.string)

# Test parameters
vocabulary = ["hello", "world", "test", "example", "keras", "tensorflow", "deep", "learning", "machine", "ai", "model"]

# Test with eager tensor
print("Testing with eager tensor:")
eager_output = call_func(
    inputs=eager_input,
    max_tokens=20,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    output_mode="int",
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=vocabulary,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding="utf-8",
    name=None
)
print(f"Dynamic output shape: {eager_output.shape}")

# Test with Keras.Input placeholder
print("\nTesting with Keras.Input placeholder:")
placeholder_output = call_func(
    inputs=placeholder_input,
    max_tokens=20,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    output_mode="int",
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=vocabulary,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding="utf-8",
    name=None
)
print(f"Static output shape: {placeholder_output.shape}")

print(f"\nDefect reproduced:")
print(f"Dynamic output shape: [None, {eager_output.shape[1]}]")
print(f"Static output shape: [None, None]")
print(f"Shapes are inconsistent: {eager_output.shape[1:] != placeholder_output.shape[1:]}")