import re
import subprocess
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import weave

# Import CodeFormatter from code_runner.py
from evalforge.code_runner import CodeFormatter
from evalforge.instructor_models import PythonAssertion, LLMAssertion
import os
import shutil

class CodeAssertionScorer(weave.Scorer):
    assertions: List[PythonAssertion]
    code_formatter: CodeFormatter = Field(default_factory=CodeFormatter)

    @weave.op()
    def score(
        self,
        model_output: Optional[Dict[str, Any]],
        input_data: Dict[str, Any],
        task_description: str,
        **kwargs
    ) -> Dict[str, Any]:
        if model_output is None or "output" not in model_output:
            return {"error": "No model output provided or 'output' key missing in model output"}

        output = model_output["output"]

        try:
            # Use the code_formatter to write assertions to files
            temp_dir = self.code_formatter.write_assertions_to_files(self.assertions)

            # Run the tests and capture the output
            test_output = self.run_tests(temp_dir, output)

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
        output_json = json.dumps({"output": output})

        # Run the test suite using subprocess and capture the output
        result = subprocess.run(
            ["python", "run_tests.py", output_json],
            capture_output=True,
            text=True,
            cwd=temp_dir,
            env=os.environ
        )
        return result.stdout + result.stderr

    def parse_test_results(self, test_output: str) -> Dict[str, Any]:
        # Extract individual test results using a more flexible regex pattern
        test_results = re.findall(r'(test_\w+).*? ... (ok|FAIL|ERROR)', test_output)

        # Initialize counts
        tests_run = len(test_results)
        passed = 0
        failures = 0
        errors = 0

        # Collect individual test results
        test_result_dict = {}
        for test_name, result in test_results:
            test_name_without_prefix = test_name  # Assuming test_name is the same as assertion.test_name
            if result == "ok":
                test_result_dict[test_name_without_prefix] = {
                    "score": 1,
                    "result": "PASS",
                    "type": "code"
                }
                passed += 1
            elif result == "FAIL":
                test_result_dict[test_name_without_prefix] = {
                    "score": 0,
                    "result": "FAIL",
                    "type": "code"
                }
                failures += 1
            elif result == "ERROR":
                test_result_dict[test_name_without_prefix] = {
                    "score": 0,
                    "result": "ERROR",
                    "type": "code"
                }
                errors += 1

        return {
            "tests_run": tests_run,
            "passed": passed,
            "failures": failures,
            "errors": errors,
            "test_results": test_result_dict
        }

def main():
    weave.init("code_evaluator_test")
    # Example Python assertions
    assertions = [
        PythonAssertion(
            test_name="within_word_limit",
            code="""
    def test_within_word_limit(self):
        total_words = len(self.output['output'].split())
        self.assertLessEqual(total_words, 150, f"Output exceeds word limit with {total_words} words.")
            """
        ),
        PythonAssertion(
            test_name="essential_information_inclusion",
            code="""
    def test_essential_information_inclusion(self):
        essential_keys = [
            'chief complaint',
            'history of present illness',
            'physical examination',
            'symptoms experienced by the patient',
            'new medications prescribed or changed',
            'follow-up instructions'
        ]
        output_text = self.output['output'].lower()
        for key in essential_keys:
            self.assertIn(key, output_text, f"Output is missing essential information: {key}.")
            """
        ),
        PythonAssertion(
            test_name="no_excessive_information",
            code="""
    def test_no_excessive_information(self):
        disallowed_terms = ['name', 'age', 'gender', 'ID']
        output_text = self.output['output'].lower()
        for term in disallowed_terms:
            self.assertNotIn(term, output_text, f"Output contains disallowed information: {term}.")
            """
        )
    ]

    # Example task description, input data, and model output
    task_description = (
        "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy "
        "guidelines and specified formatting instructions."
    )

    input_data = {
        "dialogue": (
            "Doctor: What brings you in today?\n"
            "Patient: I've been having severe headaches for the past week.\n"
            "Doctor: How often do they occur?\n"
            "Patient: Almost daily, especially in the afternoon.\n"
            "Doctor: Any other symptoms?\n"
            "Patient: I feel nauseous sometimes, and light bothers me.\n"
            "Doctor: I see. Let's do a quick examination."
        )
    }

    model_output = {
        "output": (
            "• Chief complaint: Severe headaches for the past week\n"
            "• History of present illness: The patient reports daily headaches, particularly in the afternoon, "
            "accompanied by nausea and photophobia.\n"
            "• Physical examination: Performed; details not provided.\n"
            "• Symptoms experienced by the patient: Headaches, nausea, light sensitivity.\n"
            "• New medications prescribed or changed: N/A.\n"
            "• Follow-up instructions: N/A."
        )
    }

    # Initialize the scorer with the assertions and code_formatter
    scorer = CodeAssertionScorer(
        assertions=assertions,
        code_formatter=CodeFormatter()
    )

    # Score the model output
    scores = scorer.score(model_output, input_data, task_description)

    # Print the results
    print("Code Assertion Evaluation Results:")
    print(scores)

if __name__ == "__main__":
    main()