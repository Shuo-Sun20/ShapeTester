import torch
from dataclasses import dataclass, field
from typing import List, Callable, Any

# Hook functions for the value space
def pack_hook_example(x: torch.Tensor) -> Any:
    return x.detach()

def unpack_hook_example(x: Any) -> torch.Tensor:
    return x

def pack_hook_cpu(x: torch.Tensor) -> Any:
    return x.detach().cpu()

def unpack_hook_cpu(x: Any) -> torch.Tensor:
    return x

def pack_hook_clone(x: torch.Tensor) -> Any:
    return x.detach().clone()

def unpack_hook_clone(x: Any) -> torch.Tensor:
    return x.clone()

def pack_hook_times2(x: torch.Tensor) -> Any:
    return x.detach() * 2

def unpack_hook_div2(x: Any) -> torch.Tensor:
    return x / 2

def pack_hook_abs(x: torch.Tensor) -> Any:
    return x.detach().abs()

def unpack_hook_abs(x: Any) -> torch.Tensor:
    return x

# Task 1: Define valid_test_case
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True) * 2

valid_test_case = {
    "pack_hook": pack_hook_example,
    "unpack_hook": unpack_hook_example,
    "inputs": [a, b]
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    pack_hook: List[Callable[[torch.Tensor], Any]] = field(
        default_factory=lambda: [
            pack_hook_example,
            pack_hook_cpu,
            pack_hook_clone,
            pack_hook_times2,
            pack_hook_abs
        ]
    )
    unpack_hook: List[Callable[[Any], torch.Tensor]] = field(
        default_factory=lambda: [
            unpack_hook_example,
            unpack_hook_cpu,
            unpack_hook_clone,
            unpack_hook_div2,
            unpack_hook_abs
        ]
    )