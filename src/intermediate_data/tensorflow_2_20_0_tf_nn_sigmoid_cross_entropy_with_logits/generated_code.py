import tensorflow as tf

def call_func(inputs, name=None):
    labels, logits = inputs[0], inputs[1]
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)

logits = tf.random.normal(shape=(3, 4))
labels = tf.random.uniform(shape=(3, 4), minval=0, maxval=1)
example_output = call_func(inputs=[labels, logits])