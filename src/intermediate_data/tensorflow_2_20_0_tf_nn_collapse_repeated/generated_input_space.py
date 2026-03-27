import tensorflow as tf
from dataclasses import dataclass
from typing import List, Any

valid_test_case = {
    'inputs': [tf.constant([[1, 1, 2, 2, 0, 0, 0], 
                            [2, 2, 2, 0, 0, 0, 0], 
                            [3, 4, 3, 4, 3, 4, 0]], dtype=tf.int32), 
               tf.constant([4, 3, 6], dtype=tf.int32)],
    'name': 'collapse_repeated_labels'
}

@dataclass
class InputSpace:
    inputs: List[Any] = None  # Complex structure, will be set via default_factory
    name: List[str] = None
    
    def __post_init__(self):
        if self.inputs is None:
            # Define value space for 'inputs' parameter
            # Generate 5 typical examples with varying batch sizes and sequence lengths
            batch_sizes = [1, 2, 3, 5, 8]
            max_seq_lens = [1, 3, 7, 15, 31]
            
            self.inputs = []
            for bs, msl in zip(batch_sizes, max_seq_lens):
                # Generate random labels
                labels = tf.random.uniform(
                    shape=[bs, msl], 
                    minval=0, 
                    maxval=4, 
                    dtype=tf.int32
                )
                # Generate valid sequence lengths (1 to max_seq_len)
                seq_length = tf.random.uniform(
                    shape=[bs], 
                    minval=1, 
                    maxval=msl+1, 
                    dtype=tf.int32
                )
                self.inputs.append([labels, seq_length])
        
        if self.name is None:
            # Value space for 'name' parameter
            self.name = [
                "collapse_repeated_labels", 
                "custom_name_1", 
                "test_collapse", 
                "my_collapse_op",
                "default_name"
            ]