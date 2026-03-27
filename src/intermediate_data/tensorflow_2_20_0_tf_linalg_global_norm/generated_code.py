import tensorflow as tf

def call_func(inputs, name=None):
    return tf.linalg.global_norm(t_list=inputs, name=name)

tf.random.set_seed(42)
example_input = [tf.random.normal(shape=(2, 3)), tf.random.normal(shape=(4,))]
example_output = call_func(example_input)