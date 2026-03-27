# ShapeTester Artifact (ASE Submission)

This repository contains the artifact for the paper:
**"ShapeTester: Constraint-Aware Two-Stage Test Case Generation for Detecting Shape Errors in Deep Learning Frameworks"**.

The artifact includes:
- Source code of ShapeTester.
- Intermediate inputs generated for API-level testing.
- Collected bug reports and summarized experiment results.

## Repository Layout

### data/

This directory stores experiment outputs, bug statistics, and report files.

- `all_issues.json`: Full list of detected and collected issues.
- `bug_summary.json`: Aggregated bug statistics.
- `POC_execution_results.json`: Execution results of proof-of-concept test cases.
- `shape_test_result.json`: Summary of shape testing outcomes.
- `bug_reports/`: Detailed bug reports grouped by status.
	- `concrete_bugs/`: Bugs where eager mode and graph mode produce output with different shapes.
	- `crash_bugs/`: Bugs causing crashes in either eager mode or graph mode.
- `csv_data/`: CSV-format data for research questions.
	- `RQ2/`: Data used for RQ2-related analysis.
	- `RQ3/`: Data used for RQ3-related analysis.

### src/

This directory contains the implementation of ShapeTester and supporting modules.

- `ShapeTester/main.py`: Main entry point of ShapeTester.
- `ShapeTester/GA_Based_Generation.py`: GA-based test case generation logic.
- `ShapeTester/conflict_detector.py`: Constraint conflict detection.
- `ShapeTester/inputSpace_manager.py`: Input space construction and management.
- `ShapeTester/testcase_manager.py`: Test case organization and scheduling.
- `ShapeTester/pair_manager.py`: Pairwise relation management used during generation.
- `ShapeTester/monitor.py`: Runtime monitoring utilities.
- `ShapeTester/worker.py`: Worker execution logic.
- `intermediate_data/`: Intermediate data prepared for APIs/operators under test
	(e.g., Keras/TensorFlow/PyTorch operators).
- `utils/`: Shared utilities and infrastructure code.
	- `config.py`: Configuration management.
	- `log_manager.py`: Logging support.
