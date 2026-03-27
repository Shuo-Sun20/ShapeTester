import tensorflow as tf
import numpy as np

def call_func(
    inputs,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=0,
    name="ctc_loss_dense"
):
    labels, logits = inputs[0], inputs[1]
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=logits_time_major,
        unique=unique,
        blank_index=blank_index,
        name=name
    )
    return loss

batch_size = 8
num_labels = 6
max_label_length = 5
num_frames = 12
labels = tf.random.uniform([batch_size, max_label_length], minval=1, maxval=num_labels, dtype=tf.int64)
logits = tf.random.uniform([num_frames, batch_size, num_labels])
label_length = tf.random.uniform([batch_size], minval=2, maxval=max_label_length, dtype=tf.int64)
label_mask = tf.sequence_mask(label_length, maxlen=max_label_length, dtype=label_length.dtype)
labels = labels * label_mask
logit_length = tf.convert_to_tensor([num_frames] * batch_size, dtype=tf.int32)

inputs = [labels, logits]
example_output = call_func(inputs, label_length, logit_length, blank_index=0)