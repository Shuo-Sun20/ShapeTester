import tensorflow as tf

def call_func(inputs, name=None):
    tensor = inputs[0]
    result = tf.linalg.eigvalsh(tensor=tensor, name=name)
    return result

rand_tensor = tf.random.normal(shape=(3, 3))
symmetric_tensor = 0.5 * (rand_tensor + tf.transpose(rand_tensor))
example_output = call_func([symmetric_tensor])