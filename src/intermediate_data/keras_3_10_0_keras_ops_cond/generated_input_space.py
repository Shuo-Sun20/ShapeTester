import keras
from dataclasses import dataclass, field
from typing import List, Callable

# Task 1: Define valid_test_case dictionary
def true_fn():
    return keras.random.normal(shape=(2, 3))

def false_fn():
    return keras.random.uniform(shape=(2, 3))

valid_test_case = {
    "inputs": keras.random.uniform(shape=()) > 0.5,
    "true_fn": true_fn,
    "false_fn": false_fn
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    true_fn: List[Callable] = field(default_factory=lambda: [
        lambda: keras.random.normal(shape=(2, 3)),
        lambda: keras.random.normal(shape=(3, 2)),
        lambda: keras.random.normal(shape=(4, 5)),
        lambda: keras.random.normal(shape=(1,)),
        lambda: keras.random.normal(shape=(5, 5))
    ])
    
    false_fn: List[Callable] = field(default_factory=lambda: [
        lambda: keras.random.uniform(shape=(2, 3)),
        lambda: keras.random.uniform(shape=(3, 2)),
        lambda: keras.random.uniform(shape=(4, 5)),
        lambda: keras.random.uniform(shape=(1,)),
        lambda: keras.random.uniform(shape=(5, 5))
    ])