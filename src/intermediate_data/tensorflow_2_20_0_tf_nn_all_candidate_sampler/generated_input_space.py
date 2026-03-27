import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
batch_size = 2
num_true = 3
num_sampled = 5
unique = True
seed = 0
name = None

true_classes = tf.random.uniform(
    shape=[batch_size, num_true],
    minval=0,
    maxval=num_sampled,
    dtype=tf.int64
)

valid_test_case = {
    'inputs': true_classes,
    'num_true': num_true,
    'num_sampled': num_sampled,
    'unique': unique,
    'seed': seed,
    'name': name
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    num_true: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    num_sampled: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])