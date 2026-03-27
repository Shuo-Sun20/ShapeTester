import keras
import numpy as np
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

# Create and export a simple model first
def create_and_export_model():
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    temp_dir = tempfile.mkdtemp()
    model.export(temp_dir)
    return temp_dir

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

input_tensor = np.random.random((2, 10)).astype(np.float32)

# 1. valid_test_case dictionary
valid_test_case = {
    "filepath": export_path,
    "inputs": input_tensor,
    "call_endpoint": "serve",
    "call_training_endpoint": None,
    "trainable": True,
    "name": None,
    "dtype": None,
    "training": False
}

# 2 & 3 & 4. InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    call_endpoint: List[str] = field(default_factory=lambda: ["serve", "serving_default"])
    call_training_endpoint: List[Optional[str]] = field(default_factory=lambda: [None, "serve", "serving_default", "training_endpoint"])
    training: List[bool] = field(default_factory=lambda: [True, False])
    
    # dtype parameter affects output shape only when it changes the computation graph
    dtype: List[Optional[str]] = field(default_factory=lambda: [None, "float32", "float64", "bfloat16", "float16"])
    
    # trainable parameter affects gradient computation but not output shape in forward pass
    # However, it can affect shape when model has trainable=False operations
    trainable: List[bool] = field(default_factory=lambda: [True, False])

var = InputSpace()