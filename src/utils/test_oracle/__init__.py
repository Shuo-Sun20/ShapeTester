from .keras_oracle import test_keras
from .tensorflow_oracle import test_tf
from .pytorch_oracle import test_pytorch
from .base import TestResult

def get_oracle_func(framework_name):
    if framework_name == 'torch':
        return test_pytorch
    elif framework_name == 'tensorflow':
        return test_tf
    elif framework_name == 'keras':
        return test_keras
    else:
        raise ValueError(f"Unsupported framework: {framework_name}")