import tensorflow as tf

def call_func(inputs, name=None, k=0, align="RIGHT_LEFT"):
    input_tensor = inputs[0]
    diagonal_tensor = inputs[1]
    return tf.linalg.set_diag(input=input_tensor, diagonal=diagonal_tensor, name=name, k=k, align=align)

input_tensor = tf.random.uniform(shape=(2, 3, 4))
diagonal_tensor = tf.random.uniform(shape=(2, 3))
example_output = call_func([input_tensor, diagonal_tensor])