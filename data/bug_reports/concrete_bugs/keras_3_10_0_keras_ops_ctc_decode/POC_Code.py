import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0
):
    logits = inputs
    return keras.ops.ctc_decode(
        logits,
        sequence_lengths,
        strategy=strategy,
        beam_width=beam_width,
        top_paths=top_paths,
        merge_repeated=merge_repeated,
        mask_index=mask_index
    )[0]

# Test with eager tensors (dynamic shape)
inputs_eager = np.random.randn(2, 10, 5).astype(np.float32)
sequence_lengths_eager = np.array([8, 6], dtype=np.int32)

result_eager = call_func(
    inputs_eager,
    sequence_lengths_eager,
    strategy='greedy',
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0
)

print("Dynamic output shape (eager tensors):", result_eager.shape)

# Test with Keras.Input placeholders (static shape)
inputs_placeholder = keras.Input(shape=(10, 5), batch_size=2)
sequence_lengths_placeholder = keras.Input(shape=(), batch_size=2, dtype='int32')

result_static = call_func(
    inputs_placeholder,
    sequence_lengths_placeholder,
    strategy='greedy',
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0
)

print("Static output shape (placeholders):", result_static.shape)

# Demonstrate the inconsistency
print("\nShape inconsistency detected:")
print(f"Dynamic shape: {result_eager.shape}")
print(f"Static shape: {result_static.shape}")