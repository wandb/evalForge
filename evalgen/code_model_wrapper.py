from typing import List, Dict, Optional, Any
import os
import subprocess
import json
import weave
from code_runner import CodeFormatter
from code_scorer import TestResultScorer
from weave import Evaluation
import asyncio

class CodeTestModel(weave.Model):
    assertions: dict

    @weave.op()
    def predict(self, output: dict) -> dict:
        # Initialize code formatter
        code_formatter = CodeFormatter()

        # Generate test files with the assertions
        temp_dir = code_formatter.write_assertions_to_files(self.assertions)
        # print(f"Temporary directory for test files: {temp_dir}")

        # Run the tests on the output
        result = subprocess.run(
            ["python", "run_tests.py", json.dumps(output)],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )

        # Create model output with execution result
        test_output = {
            "execution_result": {
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        }

        return test_output

async def run_code_evaluation(assertions, examples, project_name="code_evaluation_project"):
    # Initialize Weave
    weave.init(project_name)

    # Create the model
    model = CodeTestModel(assertions=assertions)

    # Create the scorer
    scorer = TestResultScorer()

    # Create the Evaluation
    evaluation = Evaluation(
        dataset=examples,
        scorers=[scorer],
    )

    # Run the evaluation
    await evaluation.evaluate(model)

if __name__ == "__main__":
    assertions = {
        "within_word_limit": """
    def test_within_word_limit(self):
        # Count words in output
        total_words = len(self.output['output'].split())
        self.assertLessEqual(total_words, 150, f"Output exceeds word limit with {total_words} words.")
    """,
        "essential_information_inclusion": """
    def test_essential_information_inclusion(self):
        # Check for the presence of essential keys
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
    """,
        "no_excessive_information": """
    def test_no_excessive_information(self):
        # Check for any mention of PII or excessive details
        disallowed_terms = ['name', 'age', 'gender', 'ID']
        output_text = self.output['output'].lower()
        for term in disallowed_terms:
            self.assertNotIn(term, output_text, f"Output contains disallowed information: {term}.")
    """
    }
    examples = [
        {
            "output": (
                "Chief complaint: The patient reports chest pain and shortness of breath.\n"
                "History of present illness: Symptoms started suddenly this morning during exercise.\n"
                "Physical examination: Irregular heartbeat detected.\n"
                "Symptoms experienced by the patient: Chest pain, dizziness.\n"
                "New medications prescribed or changed: None.\n"
                "Follow-up instructions: Return if symptoms worsen; schedule a stress test."
            )
        },
        {
            "output": (
                "The patient named John Doe, a 45-year-old male, is experiencing headaches.\n"
                "Physical examination reveals elevated blood pressure.\n"
                "New medications prescribed: Antihypertensives.\n"
                "Follow-up instructions: Monitor blood pressure daily."
            )
        },
        {
            "output": (
                "Chief complaint: Abdominal pain and nausea.\n"
                "History of present illness: Symptoms began two days ago after eating out.\n"
                "Physical examination: Tenderness in the lower abdomen.\n"
                "Symptoms experienced by the patient: Nausea, loss of appetite.\n"
                "New medications prescribed or changed: Prescribed antiemetics.\n"
                "Follow-up instructions: Rest and hydrate; follow up in one week."
            )
        }
    ]
    asyncio.run(run_code_evaluation(assertions, examples))