import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# The call_func implementation (provided in the problem)
def call_func(
    inputs,
    use_scale=True,
    dropout=0.0,
    mask=None,
    return_attention_scores=False,
    training=False,
    use_causal_mask=False
):
    layer = keras.layers.AdditiveAttention(
        use_scale=use_scale,
        dropout=dropout
    )
    if return_attention_scores:
        output, scores = layer(
            inputs=inputs,
            mask=mask,
            training=training,
            use_causal_mask=use_causal_mask,
            return_attention_scores=return_attention_scores
        )
        return output, scores
    else:
        output = layer(
            inputs=inputs,
            mask=mask,
            training=training,
            use_causal_mask=use_causal_mask,
            return_attention_scores=return_attention_scores
        )
        return output

# 1. Define valid_test_case dictionary
valid_test_case = {
    "inputs": [
        np.random.randn(2, 3, 5).astype(np.float32),  # query: (batch_size=2, Tq=3, dim=5)
        np.random.randn(2, 4, 5).astype(np.float32)   # value: (batch_size=2, Tv=4, dim=5)
    ],
    "use_scale": True,
    "dropout": 0.1,
    "mask": None,
    "return_attention_scores": False,
    "training": True,
    "use_causal_mask": False
}

# 2. Parameters that affect output shape: NONE
# The output shape of AdditiveAttention is determined solely by the input tensors' shapes:
# - query shape: (batch_size, Tq, dim) → determines batch_size, Tq
# - value shape: (batch_size, Tv, dim) → must match batch_size and dim
# Output shape: (batch_size, Tq, dim) - fixed by these inputs
# None of the other parameters (use_scale, dropout, mask, return_attention_scores, training, use_causal_mask)
# affect the shape of the main output tensor (attention output).

# 3. However, per the error feedback, we need to include all parameters that could
# potentially affect execution in InputSpace, even if they don't affect shape.
# We'll include all parameters except 'inputs' with their value spaces.

# 4. InputSpace class definition
@dataclass
class InputSpace:
    # Boolean parameters
    use_scale: List[bool] = field(default_factory=lambda: [True, False])
    training: List[bool] = field(default_factory=lambda: [True, False])
    use_causal_mask: List[bool] = field(default_factory=lambda: [True, False])
    return_attention_scores: List[bool] = field(default_factory=lambda: [True, False])
    
    # Continuous parameters (discretized to ≤5 values)
    dropout: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75])
    
    # Mask parameter (None or valid mask)
    # Note: mask can be None or a list of two boolean masks
    # For simplicity, we'll use None only in this discretization
    mask: List[Optional[List[np.ndarray]]] = field(
        default_factory=lambda: [None]  # Only None for simplicity
    )

# This class can be instantiated as: var = InputSpace()