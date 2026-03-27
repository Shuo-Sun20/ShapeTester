import torch
from torch.library import custom_op
from dataclasses import dataclass, field
from typing import Union, List, Optional

# Define the custom op and register fake impl
@custom_op("mylib::my_sin", mutates_args=())
def my_sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

@my_sin.register_fake
def my_sin_fake(x):
    return torch.empty_like(x)

# 1. valid_test_case definition
valid_test_case = {
    "op": "mylib::my_sin",
    "device_type": "cpu",
    "cast_inputs": torch.float16,
    "lib": None,
    "inputs": torch.randn(3, dtype=torch.float32, device="cpu")
}

# 2. & 3. Parameters affecting output shape (except inputs): only 'op'
# 4. InputSpace definition
@dataclass
class InputSpace:
    op: List[Union[str, torch._ops.OpOverload]] = field(
        default_factory=lambda: ["mylib::my_sin", torch.ops.mylib.my_sin]
    )