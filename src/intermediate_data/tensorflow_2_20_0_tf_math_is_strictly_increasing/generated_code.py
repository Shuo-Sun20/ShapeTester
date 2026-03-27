import tensorflow as tf

def call_func(inputs, name="is_strictly_increasing"):
    return tf.math.is_strictly_increasing(x=inputs, name=name)

# Generate a random tensor
tf.random.set_seed(42)
random_tensor = tf.random.uniform(shape=(5,), minval=0, maxval=10, dtype=tf.float32)
example_output = call_func(random_tensor)