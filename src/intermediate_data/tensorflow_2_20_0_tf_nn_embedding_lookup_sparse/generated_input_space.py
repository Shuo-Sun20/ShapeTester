import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any

# Task 1: Define valid_test_case
tf.random.set_seed(42)
params = tf.random.uniform(shape=(10, 8), minval=-1.0, maxval=1.0, dtype=tf.float32)
indices = tf.constant([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1]], dtype=tf.int64)
values = tf.constant([0, 1, 3, 2, 4], dtype=tf.int64)
dense_shape = tf.constant([3, 3], dtype=tf.int64)
sp_ids = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

valid_test_case = {
    'inputs': [params, sp_ids],
    'sp_weights': None,
    'combiner': 'sum',
    'max_norm': None,
    'name': None,
    'allow_fast_lookup': False
}

# Task 2: Identify shape-affecting parameters
# sp_weights, combiner, max_norm, name, allow_fast_lookup
# Note: None of these parameters affect output shape
# Task 3-4: Define InputSpace with parameters that don't affect shape but need testing
@dataclass
class InputSpace:
    sp_weights: List[Optional[Any]] = field(default_factory=lambda: [
        None,
        tf.SparseTensor(
            indices=indices,
            values=tf.constant([0.1, 1.0, 0.5, 1.0, 2.0], dtype=tf.float32),
            dense_shape=dense_shape
        ),
        tf.SparseTensor(
            indices=indices,
            values=tf.constant([1.0, 0.0, 0.5, 0.0, 1.0], dtype=tf.float32),
            dense_shape=dense_shape
        ),
        tf.SparseTensor(
            indices=indices,
            values=tf.constant([0.5, 0.5, 0.5, 0.5, 0.5], dtype=tf.float32),
            dense_shape=dense_shape
        ),
        tf.SparseTensor(
            indices=indices,
            values=tf.constant([2.0, 0.5, 1.0, 1.5, 0.1], dtype=tf.float32),
            dense_shape=dense_shape
        )
    ])
    
    combiner: List[str] = field(default_factory=lambda: [
        'sum',
        'mean',
        'sqrtn'
    ])
    
    max_norm: List[Optional[float]] = field(default_factory=lambda: [
        None,
        0.5,
        1.0,
        2.0,
        5.0
    ])
    
    name: List[Optional[str]] = field(default_factory=lambda: [
        None,
        'embedding_lookup',
        'test_op',
        'my_embedding',
        'lookup_sparse'
    ])
    
    allow_fast_lookup: List[bool] = field(default_factory=lambda: [
        True,
        False
    ])