import tensorflow as tf

def call_func(
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name=None
):
    weights, biases, labels, network_inputs = inputs
    return tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=network_inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        seed=seed,
        name=name
    )

batch_size = 32
dim = 128
num_classes = 10000
num_sampled = 100
num_true = 2

weights = tf.random.normal(shape=[num_classes, dim])
biases = tf.random.normal(shape=[num_classes])
labels = tf.random.uniform(
    shape=[batch_size, num_true],
    minval=0,
    maxval=num_classes,
    dtype=tf.int64
)
network_inputs = tf.random.normal(shape=[batch_size, dim])

example_output = call_func(
    inputs=[weights, biases, labels, network_inputs],
    num_sampled=num_sampled,
    num_classes=num_classes,
    num_true=num_true
)