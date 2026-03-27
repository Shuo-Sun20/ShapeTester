import tensorflow as tf

def call_func(inputs, name=None):
    return tf.nn.relu6(features=inputs, name=name)

# Generate random tensor
random_tensor = tf.random.uniform(shape=(5,), minval=-10, maxval=10, dtype=tf.float32)
example_output = call_func(inputs=random_tensor)