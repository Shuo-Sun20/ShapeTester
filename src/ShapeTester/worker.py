
from sys import is_finalizing

from preprocess.create_intermediate_dir import API_Info
from utils.test_oracle.keras_oracle import test_keras
from utils.log_manager import setup_logger
from utils.config import data_path
from inputSpace_manager import InputSpaceManager
from testcase_manager import TestCaseManager
from pair_manager import Pairwise_Tuple_Manager
from GA_Based_Generation import run_ga_based_generation
import json
from dataclasses import fields
from utils.test_oracle import TestResult
from conflict_detector import conflict_param_detect
from utils import testcase_to_str,param_equal
import argparse

logger = setup_logger(sub_dir = 'ShapeTester')
    

def get_banned_combinations(input_space_manager, conflict_params, new_test_case):
    banned_combinations = []
    for param_name in conflict_params:
        if param_name not in input_space_manager.param2int:
            logger.error(f"Conflict parameter {param_name} is not in the input space, skipping this parameter for banning.")
            return None
        if param_name not in new_test_case:
            logger.error(f"Conflict parameter {param_name} is not in the new test case: {testcase_to_str(new_test_case)} , skipping this parameter for banning.")
            return None
        param_idx = input_space_manager.param2int[param_name]
        param_value = new_test_case[param_name]
        value_idx = None
        for idx, val in enumerate(input_space_manager.int2range[param_idx]):
            if param_equal(val, param_value):
                value_idx = idx
                break
        if value_idx is None:
            logger.error(f"Conflict parameter {param_name} with value {param_value} is not in the input space range, skipping this parameter for banning.")
            return None
        banned_combinations.append((param_idx, value_idx))
    return tuple(banned_combinations)

def main(framework, version, api, api_doc, skip_list=[]):
    unit_name = f"{framework}_{version.replace('.', '_')}_{api.replace('.', '_')}"
    if unit_name in skip_list:
        logger.info(f"Skipping {unit_name} since it has been tested.")
        return "skipped"
    try:
        api_info = API_Info(framework, version, api, api_doc)
    except Exception as e:
        logger.error(f"Failed to create API info for {framework}, {version}, {api}: {e}")
        return str(e)
    logger.info(f"Successfully created intermediate data for {api_info.test_unit_name}")
    
    if not api_info.valid_testcase or not api_info.input_space:
        logger.error(f"Invalid testcase or input space for {api_info.test_unit_name}. Skipping testing.")
        return "invalid_testcase_or_input_space"
    input_space = api_info.input_space()
    input_space_manager = InputSpaceManager(input_space)
    pair_tuple_manager = Pairwise_Tuple_Manager(input_space_manager)
    testcase_manager = TestCaseManager(input_space_manager, 
                                       api_info.valid_testcase,
                                       api_info.call_func, 
                                       api_info.input_space, framework, version, api, api_doc, 
                                       api_info.oracle_func,
                                       api_info.test_unit_name)

    shape_related_parameter_list = [field.name for field in fields(input_space)]
    bug_reports = []
    pass_reports = []
    test_results = []
    initial_seeds = []
    while (True):
        # Select the best segment to test based on the current coverage status of parameter pairs
        best_segment = pair_tuple_manager.select_best_segment()
        if not best_segment:
            logger.info("All segments covered.")
            break
        new_test_case = testcase_manager.generate_test_case(best_segment)
        if new_test_case is None:
            logger.error(f"Failed to generate a new test case for segment: {best_segment}.")
            return
        
        # Execute the test case and get the test result
        test_result_info = api_info.oracle_func(new_test_case, api_info.call_func)
        test_result = test_result_info[0]
        test_results.append(str(test_result))
        pair_tuple_manager.complete_one_segment(best_segment)
        logger.info(f"Executed test case for segment: {best_segment}, result: {test_result}.")
        
        #Post-process the test result and update the pairwise tuple manager and test case manager accordingly
        if test_result == TestResult.Pass:
            #No bug found and no crash, just continue to the next segment
            logger.info("Test case executed successfully.")
            pass_reports.append({
                'test_case': testcase_to_str(new_test_case),
                'segment': str(best_segment),
                'stage': "Combinatorial Testing",
                'result': str(test_result_info)
            })
            initial_seeds.append((new_test_case, test_result_info[2][0]))
        elif test_result == TestResult.PassWithInconsistency:
            #Bug found, and no crash
            logger.warning("Inconsistency found in test case execution.")
            bug_reports.append({
                    "test_case": testcase_to_str(new_test_case),
                   'segment': str(best_segment),
                   'stage': "Combinatorial Testing",
                   'result': str(test_result_info)
                })
        else:
            #Crash happens, we need to detect the conflict parameters and ban the corresponding segments
            errorMsg = test_result_info[1]
            logger.info(f"Test case execution resulted in a crash. Error message: {errorMsg}")
            conflict_params = conflict_param_detect(framework, version, api, api_doc, new_test_case, str(errorMsg), api_info.call_func, shape_related_parameter_list,api_info.test_unit_name)
            if not conflict_params:
                logger.error(f"Error when calling function with new test case: {errorMsg}")
                continue
            else:
                logger.info(f"Conflict parameters detected: {conflict_params}")
                shape_related_conflict_params = [param for param in conflict_params if param in shape_related_parameter_list]
                shape_irrelevant_conflict_params = [param for param in conflict_params if param not in shape_related_parameter_list]
                if not shape_related_conflict_params:
                    logger.error(f"Conflict parameters detected are not related to shape: {shape_irrelevant_conflict_params}. This may indicate an issue with the conflict parameter detection.")
                elif not shape_irrelevant_conflict_params:
                    logger.info(f"Conflict parameters detected: {shape_related_conflict_params}")
                    banned_combo = get_banned_combinations(input_space_manager, shape_related_conflict_params, new_test_case)
                    if not banned_combo:
                        logger.error(f"Failed to get banned combinations for conflict parameters: {shape_related_conflict_params} with new test case: {testcase_to_str(new_test_case)}. Skipping banning for this combination.")
                        return "failed_to_get_banned_combinations"
                    pair_tuple_manager.ban_one_comb(banned_combo)
                    logger.info("Banned combination: %s", banned_combo)
                else:
                    banned_combo = {param: new_test_case[param] for param in conflict_params if param in new_test_case}
                    shape_related_conflict = testcase_manager.add_filter_condition(banned_combo, shape_related_conflict_params, best_segment)
                    if shape_related_conflict:
                        banned_combo = get_banned_combinations(input_space_manager, shape_related_conflict_params, new_test_case)
                        if not banned_combo:
                            logger.error(f"Failed to get banned combinations for conflict parameters: {shape_related_conflict_params} with new test case: {testcase_to_str(new_test_case)}. Skipping banning for this combination.")
                            return "failed_to_get_banned_combinations"
                        pair_tuple_manager.ban_one_comb(banned_combo)
                        logger.info("Banned combination: %s", banned_combo)
            #record the test result and continue to the next iteration
            if test_result == TestResult.PartialCrash:
                logger.warning("Partial crash detected, may indicate potential instability in the framework.")
                bug_reports.append({
                    'test_case': testcase_to_str(new_test_case),
                    'segment': str(best_segment),
                    'stage': "Combinatorial Testing",
                    'result': str(test_result_info),
                    'error_message': str(errorMsg),
                    'conflict_params': conflict_params
                })
            else:
                logger.warning("Crash detected, may indicate an illegal combination of parameters.")
                pass_reports.append({
                    'test_case': testcase_to_str(new_test_case),
                    'segment': str(best_segment),
                    'stage': "Combinatorial Testing",
                    'result': str(test_result_info),
                    'error_message': str(errorMsg),
                    'conflict_params': conflict_params
                })
        json.dump(bug_reports, open(api_info.work_dir / "bug_reports.json", 'w'), indent=4)
        json.dump(pass_reports, open(api_info.work_dir / "pass_reports.json", 'w'), indent=4)
        json.dump(test_results, open(api_info.work_dir / "test_results.json", 'w'), indent=4)
    logger.info("Testing completed for API: %s, framework: %s, version: %s. Total test cases executed: %d, total bugs found: %d.", api, framework, version, len(test_results), len(bug_reports))
    #Step3. Fuzzing testing: generate test cases based on the valid test case and execute the test cases.
    #Leverage Generic Algorithm to generate test cases that are similar to the valid test case but with different parameter combinations, and execute the test cases to find potential bugs.

    ga_result = run_ga_based_generation(
        initial_seeds=initial_seeds,
        input_space_manager=input_space_manager,
        oracle_func=api_info.oracle_func,
        max_generations=50,
        max_population=500,
        call_func=api_info.call_func,
        random_seed=42,
        api_info = api_info
    )
    bug_reports.extend(ga_result["bug_reports"])
    pass_reports.extend(ga_result["pass_reports"])
    test_results.extend(ga_result["test_results"])
    json.dump(bug_reports, open(api_info.work_dir / "bug_reports.json", 'w'), indent=4)
    json.dump(pass_reports, open(api_info.work_dir / "pass_reports.json", 'w'), indent=4)
    json.dump(test_results, open(api_info.work_dir / "test_results.json", 'w'), indent=4)

    if bug_reports:
        return "bug_found"
    else:
        return "no_bug_found"    

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run shape tester on a specific API.")
    args.add_argument("--framework", type=str, required=True, help="The deep learning framework to test, e.g., torch, keras, tensorflow.")
    args.add_argument("--version", type=str, required=True, help="The version of the deep learning framework to test, e.g., 2.10.0 for torch, 3.10.0 for keras, 2.20.0 for tensorflow.")
    args.add_argument("--api", type=str, required=True, help="The API to test, e.g., torch.conv1d, keras.layers.Conv1D, tf.nn.atrous_conv2d.")
    args.add_argument("--api_doc_json", type=str, required=True, help="JSON-serialized API doc used for test case generation and conflict parameter detection.") 
    args = args.parse_args()
    try:
        api_doc = json.loads(args.api_doc_json)
    except Exception:
        api_doc = args.api_doc_json

    unit_name = f"{args.framework}_{args.version.replace('.', '_')}_{args.api.replace('.', '_')}"
    try:
        result = main(args.framework, args.version, args.api, api_doc)
        print("__WORKER_RESULT__" + json.dumps({
            "unit_name": unit_name,
            "framework": args.framework,
            "version": args.version,
            "api": args.api,
            "result": result
        }, ensure_ascii=False))
    except Exception as e:
        logger.exception("Worker failed for %s", unit_name)
        print("__WORKER_RESULT__" + json.dumps({
            "unit_name": unit_name,
            "framework": args.framework,
            "version": args.version,
            "api": args.api,
            "result": f"worker_exception: {e}"
        }, ensure_ascii=False))