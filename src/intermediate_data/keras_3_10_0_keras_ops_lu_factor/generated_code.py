import keras

def call_func(inputs):
    x = inputs[0]
    lu, pivots = keras.ops.lu_factor(x)
    return [lu, pivots]

x = keras.random.normal(shape=(5, 5))
example_output = call_func([x])