import keras

def call_func(inputs, axis=0):
    return keras.ops.stack(x=inputs, axis=axis)

# Create example input tensors
tensor1 = keras.random.normal(shape=(2, 3))
tensor2 = keras.random.normal(shape=(2, 3))
tensor3 = keras.random.normal(shape=(2, 3))

example_output = call_func(inputs=[tensor1, tensor2, tensor3], axis=0)