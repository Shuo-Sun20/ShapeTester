import keras
import numpy as np

def call_func(layers, inputs, name=None, training=None):
    pipeline = keras.layers.Pipeline(layers=layers, name=name)
    output = pipeline(inputs, training=training)
    return output

# Create random input tensor
random_input = keras.random.uniform(shape=(4, 32, 32, 3), minval=0, maxval=1)

# Define layers for the pipeline
pipeline_layers = [
    keras.layers.Rescaling(scale=1./255),
    keras.layers.RandomFlip(mode="horizontal"),
    keras.layers.RandomRotation(factor=0.1),
]

# Call the function
example_output = call_func(layers=pipeline_layers, inputs=random_input)