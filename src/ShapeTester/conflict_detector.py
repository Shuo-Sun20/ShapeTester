from utils.log_manager import setup_logger
import json
from utils.llm_client.prompt_config import Conflict_Param_Detect_Config
from utils.llm_client.llm_client import run_prompt_config
import inspect
from utils import testcase_to_str

logger = setup_logger(sub_dir="conflict_detector")

def validate_helper(shape_related_parameter_list, call_func):
        def validate_func(content):
            try:
                params = json.loads(content)
                if not isinstance(params, list):
                    return "The result is not a list."
                all_parameters = list(inspect.signature(call_func).parameters.keys())
                wrong_params = [param for param in params if param not in all_parameters]
                if wrong_params:
                    return f"{wrong_params} are not valid parameters of the call_func()."
                if not any(param in shape_related_parameter_list for param in params):
                    return f"The conflict parameters identified are incorrect or incomplete. At least one of the following parameters should be included: {shape_related_parameter_list}."
                return ""
            except Exception as e:
                return str(e)
        return validate_func

def conflict_param_detect(framework_name:str, version:str, api_name:str, doc:str, new_test_case: dict, error_message: str, call_func, shape_related_parameter_list, sub_dir) -> None|list[str]:
        """
        基于错误信息，检测冲突参数。
        """
        logger.info("Detecting conflict params for test case with error: %s", error_message)
        validator = validate_helper(shape_related_parameter_list,call_func)
        Conflict_Param_Detect_Config.update_validators(validator)
        conflict_info = run_prompt_config(
                prompt_cfg=Conflict_Param_Detect_Config,
                sub_dir = sub_dir,
                use_cn=False,
                format_kwargs={"frame_name": framework_name, 
                        "version": version, 
                        "api_name": api_name,
                        "api_doc": doc,
                        "test_case": testcase_to_str(new_test_case),
                        "code_snippet": inspect.getsource(call_func),
                        "error_message": str(error_message)}
                )
        logger.info("Detected conflict parameters: %s", conflict_info['result'])
        if not conflict_info['result']:
            return None
        return json.loads(conflict_info['result'])