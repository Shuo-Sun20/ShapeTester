import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from utils.config import data_path
from utils.log_manager import setup_logger


logger = setup_logger(sub_dir='ShapeTester')
RESULT_MARKER = "__WORKER_RESULT__"


def build_ordered_tasks(raw_data):
    data = {
        'torch': ("2.10.0", raw_data['torch']),
        'keras': ("3.10.0", raw_data['keras']),
        'tensorflow': ("2.20.0", raw_data['tensorflow']),
    }

    shuffled_data = {k: [] for k in data.keys()}
    for framework, version_api_list in data.items():
        version, api_dict = version_api_list
        for api in api_dict:
            shuffled_data[framework].append((framework, version, api, api_dict[api]))

    for framework in shuffled_data.keys():
        shuffled_data[framework].sort(key=lambda x: 0 if "conv" in x[2].lower() else 1)

    max_length = max(len(api_dict) for _, api_dict in data.values())
    ordered_tasks = []
    for i in range(max_length):
        for framework in data.keys():
            if i < len(shuffled_data[framework]):
                ordered_tasks.append(shuffled_data[framework][i])
    return ordered_tasks


def parse_worker_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            payload = line[len(RESULT_MARKER):]
            return json.loads(payload)
    return None


def run_worker_task(task):
    framework, version, api, api_doc = task
    unit_name = f"{framework}_{version.replace('.', '_')}_{api.replace('.', '_')}"
    worker_script = data_path.parent / "src" / "ShapeTester" / "worker.py"
    cmd = [
        sys.executable,
        str(worker_script),
        "--framework",
        framework,
        "--version",
        version,
        "--api",
        api,
        "--api_doc_json",
        json.dumps(api_doc, ensure_ascii=False),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    parsed = parse_worker_result(completed.stdout)

    if parsed is not None:
        return parsed

    stderr_tail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
    if completed.returncode != 0:
        result = f"worker_failed_exit_{completed.returncode}"
        if stderr_tail:
            result = f"{result}: {stderr_tail}"
    else:
        result = "worker_no_result"

    return {
        "unit_name": unit_name,
        "framework": framework,
        "version": version,
        "api": api,
        "result": result,
    }


def main(max_workers=None):
    raw_data = json.load(open(data_path / 'target_API' / 'api_doc_dict.json', 'r'))
    create_result_file = data_path / 'shape_test_result.json'
    if create_result_file.exists():
        create_result = json.load(open(create_result_file, 'r'))
    else:
        create_result = {}

    skip_list = [k for k, v in create_result.items() if v]
    all_tasks = build_ordered_tasks(raw_data)
    pending_tasks = []
    for task in all_tasks:
        framework, version, api, api_doc = task
        unit_name = f"{framework}_{version.replace('.', '_')}_{api.replace('.', '_')}"
        if unit_name not in skip_list:
            pending_tasks.append((framework, version, api, api_doc))

    if not pending_tasks:
        logger.info("No pending APIs to test. All testing completed.")
        return

    if max_workers is None:
        max_workers = min(cpu_count(), len(pending_tasks))
        max_workers = max(1, max_workers)
    logger.info("Start multiprocessing monitor. pending_tasks=%d, workers=%d", len(pending_tasks), max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_worker_task, task): task for task in pending_tasks}

        for future in as_completed(future_map):
            framework, version, api, _ = future_map[future]
            unit_name = f"{framework}_{version.replace('.', '_')}_{api.replace('.', '_')}"
            try:
                result = future.result()
                create_result[result["unit_name"]] = result["result"]
                logger.info(
                    "Test result for API: %s of framework: %s, version: %s is %s.",
                    result["api"],
                    result["framework"],
                    result["version"],
                    result["result"],
                )
            except Exception as e:
                create_result[unit_name] = f"monitor_exception: {e}"
                logger.exception("Monitor failed to get future result for %s", unit_name)

            with open(create_result_file, 'w') as f:
                json.dump(create_result, f, indent=4)

    logger.info("All testing completed.")


if __name__ == "__main__":
    main(max_workers = 4)
