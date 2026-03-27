import tensorflow as tf

def call_func(inputs, shift=None, name=None):
    counts, mean_ss, variance_ss = inputs
    mean, variance = tf.nn.normalize_moments(
        counts=counts,
        mean_ss=mean_ss,
        variance_ss=variance_ss,
        shift=shift,
        name=name
    )
    return [mean, variance]

counts = tf.constant(100.0, dtype=tf.float32)
mean_ss = tf.random.normal(shape=(10,), dtype=tf.float32)
variance_ss = tf.abs(tf.random.normal(shape=(10,), dtype=tf.float32))
example_output = call_func([counts, mean_ss, variance_ss])