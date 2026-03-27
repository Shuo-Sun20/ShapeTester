import keras

def call_func(inputs):
    if isinstance(inputs, list):
        variable = inputs[0]
    else:
        variable = inputs
    return keras.ops.stop_gradient(variable)

example_tensor = keras.random.uniform(shape=(3,))
example_output = call_func(example_tensor)