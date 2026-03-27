import keras

def call_func(inputs, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.linspace(start, stop, num, endpoint, retstep, dtype, axis)

start_tensor = keras.random.uniform(shape=())
stop_tensor = keras.random.uniform(shape=())
example_output = call_func(inputs=[start_tensor, stop_tensor], num=10)