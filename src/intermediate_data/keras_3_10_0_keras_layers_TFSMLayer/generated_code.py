import keras
import numpy as np
import tempfile
import os

# Create and export a simple model first
def create_and_export_model():
    # Create a simple sequential model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Export the model to a temporary directory
    temp_dir = tempfile.mkdtemp()
    model.export(temp_dir)
    return temp_dir

# Create the model and get the export path
export_path = create_and_export_model()

def call_func(filepath, inputs, call_endpoint="serve", call_training_endpoint=None, 
              trainable=True, name=None, dtype=None, training=False):
    
    layer = keras.layers.TFSMLayer(
        filepath=filepath,
        call_endpoint=call_endpoint,
        call_training_endpoint=call_training_endpoint,
        trainable=trainable,
        name=name,
        dtype=dtype
    )
    
    output = layer(inputs, training=training)
    return output

# Generate random input tensor
input_tensor = np.random.random((2, 10)).astype(np.float32)

# Call the function and save output
example_output = call_func(
    filepath=export_path,
    inputs=input_tensor
)