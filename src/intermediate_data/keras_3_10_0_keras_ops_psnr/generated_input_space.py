import keras
from dataclasses import dataclass

# Valid test case from the example
x1 = keras.random.normal((2, 4, 4, 3))
x2 = keras.random.normal((2, 4, 4, 3))
valid_test_case = {
    "inputs": [x1, x2],
    "max_val": 1.0
}

# Parameter analysis:
# Only `max_val` affects the output tensor (scalar shape remains fixed).
# The output is always a single scalar float, but max_val affects its value.
# The value space for max_val must cover all legal continuous values > 0.

@dataclass
class InputSpace:
    # max_val: positive continuous values (discretized)
    max_val: list = None

    def __post_init__(self):
        if self.max_val is None:
            # Discretized value space for max_val:
            # - Must be > 0, typical values include small, medium, large, and boundary near 0+
            self.max_val = [
                0.1,      # Small positive boundary
                0.5,      # Typical small value
                1.0,      # Typical medium value (included from valid_test_case)
                2.0,      # Another typical value
                10.0,     # Large value
                100.0,    # Very large value
            ]