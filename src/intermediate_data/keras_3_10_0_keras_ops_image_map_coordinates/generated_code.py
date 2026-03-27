import keras

def call_func(inputs, coordinates, order, fill_mode="constant", fill_value=0):
    return keras.ops.image.map_coordinates(inputs, coordinates, order, fill_mode, fill_value)

inputs = keras.random.uniform(shape=(5, 5, 3))
coordinates = keras.random.uniform(shape=(3, 7, 7))
example_output = call_func(inputs, coordinates, order=1, fill_mode="constant", fill_value=0.0)