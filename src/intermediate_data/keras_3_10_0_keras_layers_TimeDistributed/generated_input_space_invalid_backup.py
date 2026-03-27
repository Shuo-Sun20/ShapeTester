import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union
import keras.layers as layers

# Task 1: valid_test_case dictionary
valid_test_case = {
    "layer": keras.layers.Conv2D(64, (3, 3)),
    "inputs": np.random.randn(32, 10, 128, 128, 3).astype(np.float32),
    "training": None,
    "mask": None,
    "name": None
}

# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape:
    # 1. layer: Different layer types/configurations affect output shape
    # 2. training: Some layers have different behavior in training vs inference
    # 3. mask: Masking can affect output shape if the layer supports it
    
    # Discretized value ranges (limited to ≤5 values each)
    layer: List[keras.layers.Layer] = field(default_factory=lambda: [
        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.Conv2D(64, (5, 5)),
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Dense(128),
        keras.layers.Dense(256)
    ])
    
    training: List[Optional[bool]] = field(default_factory=lambda: [
        None, True, False
    ])
    
    mask: List[Optional[np.ndarray]] = field(default_factory=lambda: [
        None,
        np.random.randint(0, 2, (32, 10), dtype=bool)
    ])
    
    name: List[Optional[str]] = field(default_factory=lambda: [
        None, "time_distributed_layer"
    ])