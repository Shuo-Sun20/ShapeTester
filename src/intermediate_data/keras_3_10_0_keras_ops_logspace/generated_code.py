import keras

def call_func(inputs, num=50, endpoint=True, base=10, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.logspace(start, stop, num, endpoint, base, dtype, axis)

start_tensor = keras.random.normal(shape=(1,))
stop_tensor = keras.random.normal(shape=(1,))
example_output = call_func([start_tensor, stop_tensor])