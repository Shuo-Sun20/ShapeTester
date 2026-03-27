"""Genetic algorithm based test case generation."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from inputSpace_manager import InputSpaceManager
from utils.log_manager import setup_logger
from utils.test_oracle import TestResult
from utils import testcase_to_str,param_equal
from utils.llm_client.prompt_config import Inputs_Extension_Config
from utils.llm_client.llm_client import run_prompt_config
from preprocess.create_intermediate_dir import API_Info
import tensorflow as tf
import torch
import keras
logger = setup_logger(sub_dir="ShapeTester")
import inspect


def _flatten_shape(shape: Any) -> list[int]:
	if shape is None:
		return [0]
	if isinstance(shape, (list, tuple)):
		values: list[int] = []
		for item in shape:
			values.extend(_flatten_shape(item))
		return values
	if isinstance(shape, bool):
		return [int(shape)]
	try:
		return [int(shape)]
	except Exception:
		return []


def _cosine_similarity(shape_a: Any, shape_b: Any) -> float:
	vec_a = _flatten_shape(shape_a)
	vec_b = _flatten_shape(shape_b)
	if len(vec_a) != len(vec_b) or not vec_a:
		return 0.0
	dot = sum(a * b for a, b in zip(vec_a, vec_b))
	norm_a = math.sqrt(sum(a * a for a in vec_a))
	norm_b = math.sqrt(sum(b * b for b in vec_b))
	if norm_a == 0 or norm_b == 0:
		return 0.0
	return dot / (norm_a * norm_b)


def _average_similarity(idx: int, population: list[tuple[dict, Any]]) -> float:
	if len(population) <= 1:
		return 0.0
	target_shape = population[idx][1]
	total = 0.0
	for j, (_, shape) in enumerate(population):
		if j == idx:
			continue
		total += math.exp(_cosine_similarity(target_shape, shape))
	return total 


def _fitness_scores(population: list[tuple[dict, Any]]) -> list[float]:
	return [1.0 / _average_similarity(i, population) for i in range(len(population))]


def _is_input_key(key: str) -> bool:
	return key.lower() in {"input", "inputs"}


def _maybe_import_torch():
	try:
		import torch  # noqa: WPS433

		return torch
	except Exception:
		return None


def _maybe_import_tf():
	try:
		import tensorflow as tf  # noqa: WPS433

		return tf
	except Exception:
		return None


def _maybe_import_numpy():
	try:
		import numpy as np  # noqa: WPS433

		return np
	except Exception:
		return None


def _shape_candidates(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
	if not shape:
		return []
	results: set[tuple[int, ...]] = set()
	for idx, dim in enumerate(shape):
		if dim <= 0:
			continue
		base = list(shape)
		if dim > 1:
			base[idx] = dim - 1
			results.add(tuple(base))
		base = list(shape)
		base[idx] = dim + 1
		results.add(tuple(base))
		base = list(shape)
		base[idx] = max(1, dim // 2)
		results.add(tuple(base))
		base = list(shape)
		base[idx] = dim * 2
		results.add(tuple(base))
	return list(results)


def _resize_tensor_torch(tensor, new_shape: tuple[int, ...]):
	torch = _maybe_import_torch()
	if torch is None:
		return None
	try:
		new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
		if tensor.numel() == 0:
			return new_tensor
		slices = tuple(slice(0, min(old, new)) for old, new in zip(tensor.shape, new_shape))
		new_tensor[slices] = tensor[slices]
		return new_tensor
	except Exception:
		return None


def _resize_tensor_tf(tensor, new_shape: tuple[int, ...]):
	tf = _maybe_import_tf()
	if tf is None:
		return None
	try:
		new_tensor = tf.zeros(new_shape, dtype=tensor.dtype)
		old_shape = tuple(int(dim) for dim in tensor.shape)
		slice_shape = [min(old, new) for old, new in zip(old_shape, new_shape)]
		if not slice_shape:
			return new_tensor
		sliced = tf.slice(tensor, [0] * len(slice_shape), slice_shape)
		paddings = [[0, new - old] for old, new in zip(slice_shape, new_shape)]
		return tf.pad(sliced, paddings)
	except Exception:
		return None


def _resize_tensor_numpy(tensor, new_shape: tuple[int, ...]):
	np = _maybe_import_numpy()
	if np is None:
		return None
	try:
		new_tensor = np.zeros(new_shape, dtype=tensor.dtype)
		slices = tuple(slice(0, min(old, new)) for old, new in zip(tensor.shape, new_shape))
		if slices:
			new_tensor[slices] = tensor[slices]
		return new_tensor
	except Exception:
		return None


def _tensor_mutations(tensor: Any, rng: random.Random) -> list[Any]:
	mutations: list[Any] = []
	shape = getattr(tensor, "shape", None)
	if shape is None:
		return mutations
	try:
		shape_tuple = tuple(int(dim) for dim in shape)
	except Exception:
		return mutations

	for new_shape in _shape_candidates(shape_tuple):
		mutated = None
		if _maybe_import_torch() is not None:
			try:
				import torch  # noqa: WPS433

				if isinstance(tensor, torch.Tensor):
					mutated = _resize_tensor_torch(tensor, new_shape)
					if mutated is not None:
						mutations.append(mutated)
						continue
			except Exception:
				pass
		if _maybe_import_tf() is not None:
			try:
				import tensorflow as tf  # noqa: WPS433

				if isinstance(tensor, tf.Tensor):
					mutated = _resize_tensor_tf(tensor, new_shape)
					if mutated is not None:
						mutations.append(mutated)
						continue
			except Exception:
				pass
		if _maybe_import_numpy() is not None:
			try:
				import numpy as np  # noqa: WPS433

				if isinstance(tensor, np.ndarray):
					mutated = _resize_tensor_numpy(tensor, new_shape)
					if mutated is not None:
						mutations.append(mutated)
						continue
			except Exception:
				pass

	if mutations:
		return mutations
	return mutations


def _mutate_inputs(test_case: dict, rng: random.Random) -> list[dict]:
	mutated_cases = [test_case]
	for key, value in test_case.items():
		if not _is_input_key(key):
			continue
		if isinstance(value, (list, tuple)):
			for idx, item in enumerate(value):
				for mutated in _tensor_mutations(item, rng):
					new_case = {k:v for k, v in test_case.items()} 
					if isinstance(value, list):
						new_value = list(value)
						new_value[idx] = mutated
					else:
						new_value = list(value)
						new_value[idx] = mutated
						new_value = tuple(new_value)
					new_case[key] = new_value
					mutated_cases.append(new_case)
		else:
			for mutated in _tensor_mutations(value, rng):
				new_case = {k:v for k, v in test_case.items()} 
				new_case[key] = mutated
				mutated_cases.append(new_case)
	return mutated_cases


def _value_equal(a: Any, b: Any) -> bool:
	return param_equal(a, b)


def _dedup_values(values: list[Any]) -> list[Any]:
	deduped: list[Any] = []
	for val in values:
		if any(_value_equal(val, existing) for existing in deduped):
			continue
		deduped.append(val)
	return deduped


def _value_signature(value: Any) -> Any:
	if hasattr(value, "shape"):
		try:
			return ("tensor", tuple(value.shape), str(getattr(value, "dtype", "")))
		except Exception:
			return ("tensor", "unknown")
	if isinstance(value, (list, tuple)):
		return tuple(_value_signature(item) for item in value)
	try:
		hash(value)
		return value
	except Exception:
		return repr(value)

def build_input_extension(api_info: API_Info|None, old_data:list[list])-> list[list]:
	if api_info is None or api_info.call_func is None or api_info.input_space is None:
		return old_data
	llm_response = run_prompt_config(Inputs_Extension_Config,sub_dir = api_info.test_unit_name, format_kwargs={
		"frame_name": api_info.framework,
		"version": api_info.version,
		"api_name": api_info.api_name,
		"api_doc": api_info.api_doc,
		"code_snippet": inspect.getsource(api_info.call_func),
		"input_space": inspect.getsource(api_info.input_space)
	})
	valid_code = llm_response.get("result", "")
	if not valid_code:
		logger.error("LLM failed to generate valid input extensions.")
		return old_data
	try:
		sandbox = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "tf": tf,
        "torch": torch,
        "keras": keras,
        }
		exec(valid_code, sandbox, sandbox)
		if "inputs_extension" not in sandbox:
			return old_data
		inputs_extension = sandbox["inputs_extension"]
		if not isinstance(inputs_extension, list):
			logger.error("LLM generated inputs_extension is not a list.")
			return old_data
		return inputs_extension
	except Exception as e:
		logger.error("Failed to execute the generated code: %s", e)
		return old_data

def _build_param_space(
	input_space_manager: InputSpaceManager,
	population: list[tuple[dict, Any]],
	api_info = None
) -> dict[str, list[Any]]:
	param_space: dict[str, list[Any]] = {}
	for param, idx in input_space_manager.param2int.items():
		param_space[param] = list(input_space_manager.int2range[idx])
	for test_case, _ in population:
		for key, value in test_case.items():
			if key not in param_space:
				param_space[key] = []
			param_space[key].append(value)
	for key, values in param_space.items():
		param_space[key] = _dedup_values(values)
	param_space['inputs'] = build_input_extension(api_info,param_space['inputs'])
	return param_space


def _select_parents(population: list[tuple[dict, Any]]) -> tuple[dict, dict] | None:
	if len(population) < 2:
		return population[0][0], population[0][0]
	scores = _fitness_scores(population)
	ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
	return population[ranked[0]][0], population[ranked[1]][0]


def _crossover(parent_a: dict, parent_b: dict, param_keys: list[str], rng: random.Random) -> list[dict]:
	if not param_keys:
		return [ {k:v for k,v in parent_a.items()}, {k:v for k,v in parent_b.items()}]
	pivot = rng.choice(param_keys)
	child_a = {k:v for k,v in parent_a.items()}
	child_b = {k:v for k,v in parent_b.items()}
	child_a[pivot], child_b[pivot] = parent_b.get(pivot), parent_a.get(pivot)
	return [child_a, child_b]


def _mutate_parameters(
	test_case: dict,
	param_space: dict[str, list[Any]],
	mutation_prob: float,
	rng: random.Random,
) -> dict:
	new_case = {k:v for k,v in test_case.items()}
	for param, values in param_space.items():
		if _is_input_key(param):
			continue
		if not values:
			continue
		if rng.random() < mutation_prob:
			new_case[param] = rng.choice(values)
	return new_case


def _test_case_fingerprint(test_case: dict) -> tuple:
	items = []
	for key in sorted(test_case.keys()):
		value = test_case[key]
		items.append((key, _value_signature(value)))
	return tuple(items)


def run_ga_based_generation(
	initial_seeds: list[tuple[dict, Any]],
	input_space_manager: InputSpaceManager,
	oracle_func,
	call_func,
	max_generations: int = 8,
	mutation_prob: float = 0.2,
	input_mutation_prob: float = 0.5,
	offspring_per_generation: int = 2,
	max_population: int = 200,
	random_seed: int | None = None,
	api_info = None,
) -> dict[str, Any]:
	if not initial_seeds:
		logger.info("No initial seeds available for GA generation.")
		return {"bug_reports": [], "pass_reports": [], "test_results": [], "population": []}

	rng = random.Random(random_seed)
	population: list[tuple[dict, Any]] = [seed for seed in initial_seeds if seed[1] is not None]
	if len(population) < 2:
		logger.info("Not enough seeds for GA generation.")
		return {"bug_reports": [], "pass_reports": [], "test_results": [], "population": population}

	bug_reports: list[dict] = []
	pass_reports: list[dict] = []
	test_results: list[str] = []
	seen_cases = {_test_case_fingerprint(tc) for tc, _ in population}

	for generation in range(max_generations):
		param_space = _build_param_space(input_space_manager, population,api_info)
		parent_pair = _select_parents(population)
		if not parent_pair:
			break
		parent_a, parent_b = parent_pair
		param_keys = [k for k in param_space.keys() if not _is_input_key(k)]
		offspring = []
		for _ in range(offspring_per_generation):
			children = _crossover(parent_a, parent_b, param_keys, rng)
			for child in children:
				new_child = False
				mutated = None
				for retry_time in range(3):
					mutated = _mutate_parameters(child, param_space, mutation_prob+retry_time*0.1, rng)
					fingerprint = _test_case_fingerprint(mutated)
					if fingerprint not in seen_cases:
						new_child = True
						seen_cases.add(fingerprint)
						break
				if new_child:
					offspring.append(mutated)

		expanded_offspring: list[dict] = []
		for child in offspring:
			expanded_offspring.append(child)
   
		logger.info("GA generation %d: %d offspring generated, %d after mutation and deduplication.", generation + 1, len(offspring), len(expanded_offspring))
		for test_case in expanded_offspring:
			test_result_info = oracle_func(test_case, call_func)
			test_result = test_result_info[0]
			test_results.append(str(test_result))

			if test_result == TestResult.Pass:
				pass_reports.append({"test_case": testcase_to_str(test_case), "stage": "Genetic Algorithm", "result": str(test_result_info)})
				population.append((test_case, test_result_info[2][0]))
			elif test_result == TestResult.PassWithInconsistency:
				bug_reports.append({"test_case": testcase_to_str(test_case), "stage": "Genetic Algorithm", "result": str(test_result_info)})
			elif test_result == TestResult.PartialCrash:
				bug_reports.append({
					"test_case": testcase_to_str(test_case),
					"stage": "Genetic Algorithm",
					"result": str(test_result_info),
					"error_message": str(test_result_info[1]),
				})
			else:
				pass_reports.append({
					"test_case": testcase_to_str(test_case),
					"stage": "Genetic Algorithm",
					"result": str(test_result_info),
					"error_message": str(test_result_info[1]),
				})

			if len(population) > max_population:
				population = population[-max_population:]

		logger.info(
			"GA generation %d completed. Population size: %d", generation + 1, len(population)
		)

	return {
		"bug_reports": bug_reports,
		"pass_reports": pass_reports,
		"test_results": test_results,
		"population": population,
	}
