import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import weave
from pydantic import Field

# Import CodeFormatter from code_runner.py
from evalforge.utils import logger
from evalforge.code_formatter import CodeFormatter
from evalforge.instructor_models import PythonAssertion


class CodeAssertionScorer(weave.Scorer):
    assertions: List[PythonAssertion]
    code_formatter: CodeFormatter = Field(default_factory=CodeFormatter)

    @weave.op
    def score(
        self,
        model_output: Optional[Dict[str, Any]],
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        if model_output is None:
            logger.error("No model output provided")
            return {"code_assertion_results": {
                "tests_run": 0,
                "passed": 0,
                "failures": 0,
                "errors": 0,
                "test_results": {}
            }}

        try:
            # Use the code_formatter to write assertions to files
            temp_dir = self.code_formatter.write_assertions_to_files(self.assertions)

            # Run the tests and capture the output
            test_output = self.run_tests(temp_dir, model_output)

            # Parse the test results to extract scores
            scores = self.parse_test_results(test_output)
            scores["raw_output"] = test_output  # Include raw test output for reference

            return {"code_assertion_results": scores}
        finally:
            # Delete the temporary directory
            if temp_dir:
                shutil.rmtree(temp_dir)

    def run_tests(self, temp_dir: str, output: Any) -> str:
        import json

        # Create a test context with both the model output and input data
        test_context = json.dumps({
            "output": output,
            "input": output.get("input_data", {}) if isinstance(output, dict) else {}
        })

        # Run the test suite using subprocess and capture the output
        result = subprocess.run(
            ["python", "run_tests.py", test_context],
            capture_output=True,
            text=True,
            cwd=temp_dir,
            env=os.environ,
        )
        return result.stdout + result.stderr

    def parse_test_results(self, test_output: str) -> Dict[str, Any]:
        # Extract individual test results using a more flexible regex pattern
        test_results = re.findall(r"(test_\w+).*? ... (ok|FAIL|ERROR)", test_output)

        # Initialize counts
        tests_run = len(test_results)
        passed = 0
        failures = 0
        errors = 0

        # Collect individual test results
        test_result_dict = {}
        for test_name, result in test_results:
            test_name_without_prefix = (
                test_name  # Assuming test_name is the same as assertion.test_name
            )
            if result == "ok":
                test_result_dict[test_name_without_prefix] = {
                    "score": 1,
                    "result": "PASS",
                    "type": "code",
                }
                passed += 1
            elif result == "FAIL":
                test_result_dict[test_name_without_prefix] = {
                    "score": 0,
                    "result": "FAIL",
                    "type": "code",
                }
                failures += 1
            elif result == "ERROR":
                test_result_dict[test_name_without_prefix] = {
                    "score": 0,
                    "result": "ERROR",
                    "type": "code",
                }
                errors += 1

        return {
            "tests_run": tests_run,
            "passed": passed,
            "failures": failures,
            "errors": errors,
            "test_results": test_result_dict,
        }