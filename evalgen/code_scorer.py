import re
from typing import Dict, Any, Optional
import weave

class TestResultScorer(weave.Scorer):
    @weave.op()
    def parse_test_results(self, test_output: str) -> Dict[str, int]:
        # Extract the summary line
        summary_match = re.search(
            r"Ran (\d+) tests? in \d+\.\d+s\n\n(OK|FAILED \(failures=(\d+)(?:, errors=(\d+))?\))",
            test_output
        )
        if not summary_match:
            return {"error": "Could not parse test summary"}

        tests_run = int(summary_match.group(1))
        if summary_match.group(2) == "OK":
            passed = tests_run
            failures = 0
            errors = 0
        else:
            failures = int(summary_match.group(3))
            errors = int(summary_match.group(4)) if summary_match.group(4) else 0
            passed = tests_run - (failures + errors)

        return {
            "tests_run": tests_run,
            "passed": passed,
            "failures": failures,
            "errors": errors
        }

    @weave.op()
    def score(self, model_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if model_output is None or "execution_result" not in model_output:
            return {"error": "No execution result provided in model output"}

        execution_result = model_output["execution_result"]
        test_output = execution_result.get("stdout", "") + "\n" + execution_result.get("stderr", "")

        scores = self.parse_test_results(test_output)
        scores["raw_output"] = test_output

        return scores

def main():
    # Usage example:
    scorer = TestResultScorer()

    # Example model output
    model_output = {
        "execution_result": {
            "stdout": """test_essential_information_inclusion (tests.test_essential_information_inclusion.TestEssential_information_inclusion.test_essential_information_inclusion) ... ok
    test_no_excessive_information (tests.test_no_excessive_information.TestNo_excessive_information.test_no_excessive_information) ... FAIL
    test_within_word_limit (tests.test_within_word_limit.TestWithin_word_limit.test_within_word_limit) ... FAIL

    ======================================================================
    FAIL: test_no_excessive_information (tests.test_no_excessive_information.TestNo_excessive_information.test_no_excessive_information)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
    File "/path/to/tests/test_no_excessive_information.py", line 12, in test_no_excessive_information
        self.assertNotIn(term, output_text, f"Output contains disallowed information: {term}.")
    AssertionError: 'age' unexpectedly found in '...' : Output contains disallowed information: age.

    ======================================================================
    FAIL: test_within_word_limit (tests.test_within_word_limit.TestWithin_word_limit.test_within_word_limit)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
    File "/path/to/tests/test_within_word_limit.py", line 10, in test_within_word_limit
        self.assertLessEqual(total_words, 150, f"Output exceeds word limit with {total_words} words.")
    AssertionError: 173 not less than or equal to 150 : Output exceeds word limit with 173 words.

    ----------------------------------------------------------------------
    Ran 3 tests in 0.001s

    FAILED (failures=2)
    """,
            "stderr": ""
        }
    }

    # Score the model output
    scores = scorer.score(model_output)
    print("Test Results and Scores:")
    print(scores)

if __name__ == "__main__":
    main()