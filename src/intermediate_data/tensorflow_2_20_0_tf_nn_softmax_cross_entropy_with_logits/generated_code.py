import tensorflow as tf

def call_func(inputs, axis=-1, name=None):
    labels, logits = inputs
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, axis=axis, name=name)

logits = tf.random.uniform(shape=(2, 3), minval=-1, maxval=1, dtype=tf.float32)
labels = tf.random.uniform(shape=(2, 3), minval=0, maxval=1, dtype=tf.float32)
labels = labels / tf.reduce_sum(labels, axis=1, keepdims=True)
example_output = call_func(inputs=[labels, logits])