from collections.abc import Callable
from enum import Enum


class TestResult(Enum):
    AllCrash = "AllCrash"
    PartialCrash = "PartialCrash"
    PassWithInconsistency = "PassWithInconsistency"
    Pass = "Pass"

def get_output_shape(output):
    if hasattr(output, 'shape'):
        return tuple(output.shape)
    elif isinstance(output, (list, tuple)):
        return tuple(get_output_shape(o) for o in output)
    else:
        return None

def as_oracle(specific_test_func):
    def wrapper(test_case:dict, test_func:Callable):
        result = specific_test_func(test_case, test_func)
        ## result should be a list of results, [eager_result, symbolic_result, [meta_result]]
        errorMsg = None
        test_result = None
        for res_info in result:
            if isinstance(res_info, str):
                errorMsg = res_info
                break
            elif res_info is None:
                errorMsg = ""
        if errorMsg is None:
            eager_shape = result[0]
            for res_info in result[1:]:
                if res_info != eager_shape:
                    test_result = TestResult.PassWithInconsistency
                    break
            if test_result is None:
                test_result = TestResult.Pass
        else:
            for res_info in result:
                if not (isinstance(res_info, str) or res_info is None):
                    test_result = TestResult.PartialCrash
                    break
            if test_result is None:
                test_result = TestResult.AllCrash
        return test_result, errorMsg, result
    return wrapper
            