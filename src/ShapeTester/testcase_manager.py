
from hmac import new

from numpy import equal, isin

from inputSpace_manager import InputSpaceManager
from utils.log_manager import setup_logger
from utils.llm_client.llm_client import run_prompt_config
from utils.llm_client.prompt_config import Test_Case_Complete_Config
import json
import inspect
from utils import testcase_to_str,param_equal
import tensorflow as tf
import torch
import keras

logger = setup_logger(sub_dir="core")

def generate_test_case_with_LLM(input_space, partial_test_case, call_func, framework, version, api, doc, test_body_validator, sub_dir) -> dict|None|list:
    def new_validator(code):
        sandbox = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "tf": tf,
        "torch": torch,
        "keras": keras,
        }

        try:
            exec(code, sandbox, sandbox)
            if "test_case" not in sandbox:
                raise Exception("test_case variable not defined.")
            test_case = sandbox["test_case"]
            if not isinstance(test_case, dict):
                raise Exception("test_case is not a dict.")
            errorMsg = test_body_validator(test_case)
            if errorMsg:
                raise Exception(f"Generated test case failed during execution: {errorMsg}")
        except Exception as e:  # noqa: BLE001
            exec_errMsg = str(e)
            try:
                data = json.loads(code)
                if isinstance(data, list):
                    return ""
                else:
                    raise Exception(f"Generated content is not a list: {data}")
            except Exception as json_e:
                json_errMsg = str(json_e)
                return f"Failed to execute the generated code: {exec_errMsg}. Additionally, failed to parse the generated content as JSON: {json_errMsg}."
        return ""
    Test_Case_Complete_Config.result_validator = new_validator  

    llm_response = run_prompt_config(Test_Case_Complete_Config, sub_dir = sub_dir, format_kwargs={
        "input_space": input_space,
        "frame_name": framework,
        "version": version,
        "api_name": api,
        "api_doc": doc,
        "partial_test_case": partial_test_case,
        "code_snippet": inspect.getsource(call_func)
    })
    valid_code = llm_response.get("result", "")
    if not valid_code:
        logger.error("LLM failed to generate a valid test case.")
        return None
    try:
        data = json.loads(valid_code)
        if not isinstance(data, list):
            logger.error("LLM generated test case is not a list.")
            return None
        logger.info(f"found conflict: {data}")
        return data
    except Exception as e:
        logger.info("No conflict, trying to execute the generated code.")
        sandbox = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "tf": tf,
        "torch": torch,
        "keras": keras,
        }
        try:
            exec(valid_code, sandbox, sandbox)
            if "test_case" not in sandbox:
                return None
            test_case = sandbox["test_case"]
            if not isinstance(test_case, dict):
                return None
            return test_case
        except Exception as e:
            logger.error("Failed to execute the generated code: %s", e)
            return None

def get_validate(oracle_func, call_func):
    def validate_func(test_case):
        test_result_info = oracle_func(test_case, call_func)
        test_result = test_result_info[0]
        errorMsg = test_result_info[1]
        if not errorMsg:
            return ""
        else:
            return f"Test case execution failed: {errorMsg}"
    return validate_func


class TestCaseManager:
    def __init__(self, input_space_manager:InputSpaceManager, valid_test_case:dict, call_func, InputSpace, framework, version, api, doc, oracle_func, sub_dir):
        self.input_space_manager:InputSpaceManager = input_space_manager
        self.valid_test_case_list:list[dict] = [valid_test_case] 
        self.banned_combinations:list[dict] = []
        self.call_func = call_func
        self.framework = framework
        self.version = version
        self.Input_Space = InputSpace
        self.api = api
        self.doc = doc
        self.sub_dir = sub_dir
        self.test_body_validator = get_validate(oracle_func, call_func)
        self.suitable_combo_list = [{}]  # List of dicts, each dict is a combination of parameters that has been tested and found to be suitable (no crash)
        logger.info("TestCaseManager initialized with 1 valid test case.")
    
    def generate_test_case(self, segment:tuple[int,...]) -> dict|None:
        partial_test_case = self.input_space_manager.segment_to_testcase(segment)
        logger.debug("Generating test case for segment: %s", testcase_to_str(partial_test_case))
        tc_candidates = []
        for tc_cnt,valid_tc in enumerate(self.valid_test_case_list):
            is_compatible = True
            new_tc = valid_tc.copy()
            new_tc.update(partial_test_case)
            for banned_combo in self.banned_combinations:
                if all([param_equal(new_tc.get(param), value) for param, value in banned_combo.items()]):
                    is_compatible = False
                    break
            if is_compatible:
                logger.debug("Compatible test case found.")
                score = 0
                for param, value in self.suitable_combo_list[tc_cnt].items():
                    if param_equal(partial_test_case.get(param), value):
                        score += 1
                logger.debug(f"Evaluating compatibility score for test case: {testcase_to_str(partial_test_case)} against suitable combination: {testcase_to_str(self.suitable_combo_list[tc_cnt])}, score: {score}")
                tc_candidates.append((new_tc, score))
        if tc_candidates:
            best_tc = max(tc_candidates, key=lambda x: x[1])[0]
            best_tc_score = max(tc_candidates, key=lambda x: x[1])[1]
            logger.debug(f"Best compatible test case: {testcase_to_str(best_tc)} with score: {best_tc_score}")
            new_tc = best_tc
            return new_tc
        # If no compatible test case found, generate a new one with LLM
        logger.error("No compatible test case found.")
        return None
    
    def generate_partial_test_case(self, shape_related_conflict_params:list[str], best_segment) -> str:
        partial_test_case = {
            param: best_segment[self.input_space_manager.param2int[param]] for param in shape_related_conflict_params
        }
        partial_code = """
```python
input_space = Input_Space()
pre_defined = {\n
"""+',\n'.join([f'"{param}": input_space.{param}[{value}]' for param, value in partial_test_case.items()]) + "\n}```"
        return partial_code
    
    def add_filter_condition(self, condition_func, shape_related_conflict_params, best_segment):
        new_test_case = generate_test_case_with_LLM(
            inspect.getsource(self.Input_Space),
            self.generate_partial_test_case(shape_related_conflict_params,best_segment),
            self.call_func,
            self.framework,
            self.version,
            self.api,
            self.doc,
            self.test_body_validator,
            self.sub_dir
        )

        if not new_test_case:
            logger.error("Failed to generate a new test case with LLM, cannot add filter condition.")
            return
        elif isinstance(new_test_case, list):
            logger.info("LLM found new conflict parameters: %s", new_test_case)
            return new_test_case
        elif isinstance(new_test_case, dict):
            logger.info("LLM generated a new valid test case: %s", testcase_to_str(new_test_case))
            logger.info("Added filter condition: %s", testcase_to_str(condition_func))
            self.banned_combinations.append(condition_func)
            partial_test_case = {param: condition_func[param] for param in shape_related_conflict_params}
            self.suitable_combo_list.append(partial_test_case)
            self.valid_test_case_list.append(new_test_case)
        else:
            logger.error("LLM returned an unexpected format: %s", testcase_to_str(new_test_case))