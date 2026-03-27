import keras.ops
import keras

def call_func(inputs, axis=-1, order=2, epsilon=None):
    return keras.ops.normalize(x=inputs, axis=axis, order=order, epsilon=epsilon)

x = keras.random.uniform(shape=(2, 3, 4))
example_output = call_func(x)