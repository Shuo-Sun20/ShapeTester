import keras
from keras import ops

def call_func(inputs, default=0):
    condlist = inputs[0]
    choicelist = inputs[1]
    return ops.select(condlist, choicelist, default)

x = keras.random.uniform(shape=(6,))
condlist = [x < 0.3, x > 0.7]
choicelist = [x, x**2]
inputs = [condlist, choicelist]
example_output = call_func(inputs, default=42)