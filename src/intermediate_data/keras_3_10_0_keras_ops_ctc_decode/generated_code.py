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

batch_size = 2
max_length = 10
num_classes = 5
logits = np.random.randn(batch_size, max_length, num_classes).astype(np.float32)
sequence_lengths = np.random.randint(1, max_length + 1, size=(batch_size,)).astype(np.int32)

example_output = call_func(logits, sequence_lengths, strategy="greedy")