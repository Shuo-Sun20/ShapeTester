import tensorflow as tf

def call_func(inputs):
    return tf.nn.scale_regularization_loss(inputs)

weights = tf.random.normal(shape=(10, 5), dtype=tf.float32)
regularization_loss = tf.nn.l2_loss(weights)
example_output = call_func(regularization_loss)