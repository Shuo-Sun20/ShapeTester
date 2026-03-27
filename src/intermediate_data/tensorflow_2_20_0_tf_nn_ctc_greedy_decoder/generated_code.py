import tensorflow as tf

def call_func(inputs, sequence_length, merge_repeated=True, blank_index=None):
    # inputs is expected to be a single tensor for logits
    # sequence_length is separate parameter as per API
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs,
        sequence_length,
        merge_repeated=merge_repeated,
        blank_index=blank_index
    )
    return [decoded[0], neg_sum_logits]

# Generate random inputs
max_time = 5
batch_size = 2
num_classes = 6
seq_lens = tf.constant([3, 5], dtype=tf.int32)
logits = tf.random.normal(shape=[max_time, batch_size, num_classes])

# Call the function
example_output = call_func(logits, seq_lens)