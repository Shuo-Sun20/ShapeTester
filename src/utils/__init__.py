

def param_to_str(param):
    if isinstance(param, (list, tuple) ):
        return "[" + ", ".join(param_to_str(p) for p in param) + "]"
    elif isinstance(param, dict):
        return "{" + ", ".join(f"{k}: {param_to_str(v)}" for k, v in param.items()) + "}"
    elif hasattr(param, 'shape'):
        return f"{type(param).__name__}(shape={param.shape})"
    else:
        return str(param)
    
def testcase_to_str(test_case):
    test_case_str = {
        k: param_to_str(v) for k, v in test_case.items()
    }
    return test_case_str

def param_equal(param1, param2):
    if type(param1) != type(param2):
        return False
    if hasattr(param1, 'shape') and hasattr(param2, 'shape'):
        return param1.shape == param2.shape
    elif isinstance(param1, list) and isinstance(param2, list):
        if len(param1) != len(param2):
            return False
        return all(param_equal(p1, p2) for p1, p2 in zip(param1, param2))
    elif isinstance(param1, tuple) and isinstance(param2, tuple):
        if len(param1) != len(param2):
            return False
        return all(param_equal(p1, p2) for p1, p2 in zip(param1, param2))
    else:
        return param1 == param2
