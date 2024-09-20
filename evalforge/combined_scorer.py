import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import weave
from pydantic import Field

from evalforge.code_evaluator import CodeAssertionScorer, CodeFormatter
from evalforge.instructor_models import (Criterion, LLMAssertion,
                                         PythonAssertion)
from evalforge.llm_evaluator import LLMAssertionScorer


@weave.op()
def predict_passthrough(
    model_output: Dict[str, Any], task_description: str, input_data: Dict[str, Any]
) -> Dict[str, Any]:
    return model_output


from typing import Any, Dict

import weave

from evalforge.criterion_assertion_map import CriterionAssertionMap


class AssertionScorer(weave.Scorer):
    criterion_assertion_map: CriterionAssertionMap = Field(
        default_factory=CriterionAssertionMap
    )
    llm_model: str = Field(default="gpt-4o-2024-08-06")
    prompt_template: str = Field(
        default="""
Task Description:
{task_description}

Evaluate the following output based on the given task, input, and assertion:

Input:
{input_data}

Output:
{model_output}

Assertion:
{assertion_text}

Consider the task description and input when evaluating the output against the assertion.
Respond with either 'PASS' if the output meets the assertion criteria in the context of the task and input, or 'FAIL' if it does not.
"""
    )
    system_prompt: str = Field(
        default="You are an AI assistant evaluating the quality of text outputs based on given tasks, inputs, and assertions."
    )
    code_formatter: CodeFormatter = Field(default_factory=CodeFormatter)

    def get_grouped_assertions_by_type(self):
        # Collect all assertions from the mapping
        all_assertions = []
        for assertions in self.criterion_assertion_map.criterion_to_assertions.values():
            all_assertions.extend(assertions)

        # Separate assertions into LLM and Python assertions
        llm_assertions = [a for a in all_assertions if isinstance(a, LLMAssertion)]
        python_assertions = [
            a for a in all_assertions if isinstance(a, PythonAssertion)
        ]
        return llm_assertions, python_assertions

    @weave.op()
    async def score(
        self,
        model_output: Optional[Dict[str, Any]],
        task_description: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if model_output is None:
            return {"error": "No model output provided"}

        results = {}

        llm_assertions, python_assertions = self.get_grouped_assertions_by_type()

        # Process LLM assertions
        if llm_assertions:
            llm_scorer = LLMAssertionScorer(
                assertions=llm_assertions,
                model=self.llm_model,
                prompt_template=self.prompt_template,
                system_prompt=self.system_prompt,
            )
            llm_results = await llm_scorer.score(
                model_output, task_description, input_data
            )
            results["llm_assertion_results"] = llm_results.get(
                "llm_assertion_results", {}
            )

        # Process Python assertions
        if python_assertions:
            code_scorer = CodeAssertionScorer(
                assertions=python_assertions,
                code_formatter=self.code_formatter,
            )
            code_results = code_scorer.score(model_output, input_data, task_description)
            results["code_assertion_results"] = code_results.get(
                "code_assertion_results", {}
            )

        # Map results back to criteria using the mapping class
        criterion_results: Dict[str, Dict[str, Any]] = {}
        for test_name, result in results.get("llm_assertion_results", {}).items():
            criterion = self.criterion_assertion_map.get_criterion_by_assertion(
                test_name
            )
            if criterion not in criterion_results:
                criterion_results[criterion] = {}
            criterion_results[criterion][test_name] = result

        for test_name, result in (
            results.get("code_assertion_results", {}).get("test_results", {}).items()
        ):
            criterion = self.criterion_assertion_map.get_criterion_by_assertion(
                test_name
            )
            if criterion not in criterion_results:
                criterion_results[criterion] = {}
            criterion_results[criterion][test_name] = result

        return criterion_results

    def export(self, base_dir: str = "forged_judge"):
        base_dir = Path(base_dir)
        llm_dir = base_dir / "llm_assertions"
        python_dir = base_dir / "python_assertions"

        base_dir.mkdir(parents=True, exist_ok=True)
        llm_dir.mkdir(parents=True, exist_ok=True)
        python_dir.mkdir(parents=True, exist_ok=True)

        llm_assertions, python_assertions = self.get_grouped_assertions_by_type()

        # Export LLM Assertions
        self.export_assertions_by_criteria(llm_assertions, llm_dir)

        # Export Python Assertions
        self.export_assertions_by_criteria(python_assertions, python_dir)

    def export_assertions_by_criteria(self, assertions, base_dir: Path):
        # Save assertions in subfolders matching their criteria
        for assertion in assertions:
            criterion = self.criterion_assertion_map.get_criterion_by_assertion(
                assertion.test_name
            )
            if criterion is None:
                criterion = "unknown_criterion"
            criterion_dir = base_dir / criterion
            criterion_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(assertion, LLMAssertion):
                filename = f"{assertion.test_name}.txt"
                file_path = criterion_dir / filename
                file_path.write_text(assertion.text, encoding="utf-8")
            elif isinstance(assertion, PythonAssertion):
                filename = f"{assertion.test_name}.py"
                file_path = criterion_dir / filename
                file_path.write_text(assertion.code, encoding="utf-8")

    def import_assertions(self, base_dir: str = "forged_judge"):
        base_dir = Path(base_dir)
        llm_dir = base_dir / "llm_assertions"
        python_dir = base_dir / "python_assertions"

        # Load LLM Assertions
        llm_assertions = self.load_assertions_by_criteria(llm_dir, LLMAssertion)

        # Load Python Assertions
        python_assertions = self.load_assertions_by_criteria(
            python_dir, PythonAssertion
        )

        # Clear existing mappings
        self.criterion_assertion_map.criterion_to_assertions = {}
        self.criterion_assertion_map.assertion_to_criterion = {}

        # Update the criterion_assertion_map
        for item in llm_assertions + python_assertions:
            criterion = item["criterion"]
            assertion = item["assertion"]
            self.criterion_assertion_map.add_assertion(
                Criterion(criterion=criterion), assertion
            )

    def load_assertions_by_criteria(self, base_dir: Path, assertion_cls):
        assertions = []
        for criterion_dir in base_dir.iterdir():
            if criterion_dir.is_dir():
                criterion = criterion_dir.name
                for file in criterion_dir.iterdir():
                    if file.is_file():
                        test_name = file.stem
                        content = file.read_text(encoding="utf-8")

                        if assertion_cls == LLMAssertion:
                            assertion = LLMAssertion(
                                test_name=test_name, text=content, evaluation_type="llm"
                            )
                        elif assertion_cls == PythonAssertion:
                            assertion = PythonAssertion(
                                test_name=test_name,
                                code=content,
                                evaluation_type="python",
                            )
                        else:
                            continue

                        assertions.append(
                            {"criterion": criterion, "assertion": assertion}
                        )
        return assertions


async def main():
    import weave

    weave.init("combined_scorer_test")

    # Example assertions (combining both LLM and Python assertions)
    assertions = [
        # LLM Assertions
        LLMAssertion(
            test_name="accuracy_and_completeness_evaluation",
            text=(
                "Evaluate the provided medical note summary and determine if it accurately and completely "
                "captures all relevant information from the original dialogue.\n"
                "Ensure the following sections are covered comprehensively and without errors:\n"
                "- Chief complaint\n"
                "- History of present illness\n"
                "- Physical examination findings\n"
                "- Symptoms experienced\n"
                "- New medications prescribed or changed with correct dosages\n"
                "- Follow-up instructions\n"
                "The information should be free of personal identifiable information (PII), and the use of 'N/A' "
                "should be correctly applied when there is no applicable information. The format should adhere to "
                "bullet points starting with the key. Based on this assessment, respond with 'PASS' if all criteria "
                "are met, otherwise 'FAIL'."
            ),
        ),
        LLMAssertion(
            test_name="conciseness_and_privacy_compliance",
            text=(
                "Evaluate the following output for conciseness and privacy compliance: Does the output summarize the "
                "key information effectively within 150 words while ensuring no personal identifiable information (PII) "
                "like name, age, gender, or ID is present? Provide your assessment as PASS for compliance or FAIL otherwise."
            ),
        ),
        # Python Assertions
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
            """,
        ),
        PythonAssertion(
            test_name="no_excessive_information",
            code="""
    def test_no_excessive_information(self):
        disallowed_terms = ['name', 'age', 'gender', 'id']
        output_text = self.output['output'].lower()
        for term in disallowed_terms:
            self.assertNotIn(term, output_text, f"Output contains disallowed information: {term}.")
            """,
        ),
    ]

    # Examples
    examples = [
        # Example 1
        {
            "task_description": (
                "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy "
                "guidelines and specified formatting instructions."
            ),
            "input_data": {
                "dialogue": (
                    "Doctor: What brings you in today?\n"
                    "Patient: I've been having severe headaches for the past week.\n"
                    "Doctor: How often do they occur?\n"
                    "Patient: Almost daily, especially in the afternoon.\n"
                    "Doctor: Any other symptoms?\n"
                    "Patient: I feel nauseous sometimes, and light bothers me.\n"
                    "Doctor: I see. Let's do a quick examination."
                )
            },
            "model_output": {
                "output": (
                    "• Chief complaint: Severe headaches for the past week\n"
                    "• History of present illness: The patient reports daily headaches, particularly in the afternoon, accompanied "
                    "by nausea and photophobia.\n"
                    # Omitted 'Physical examination' to induce failure
                    # "• Physical examination: Performed; details not provided.\n"
                    "• Symptoms experienced by the patient: Headaches, nausea, light sensitivity.\n"
                    "• New medications prescribed or changed: N/A.\n"
                    "• Follow-up instructions: N/A."
                )
            },
            # Expected outcomes:
            # - LLM Assertions: One PASS, one FAIL
            # - Code Assertions: One PASS, one FAIL
        },
        # Example 2
        {
            "task_description": (
                "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy "
                "guidelines and specified formatting instructions."
            ),
            "input_data": {
                "dialogue": (
                    "Doctor: How are you feeling today?\n"
                    "Patient: I've had a persistent cough and fever for the last three days.\n"
                    "Doctor: Have you noticed any other symptoms?\n"
                    "Patient: Just some fatigue, and occasionally I feel short of breath.\n"
                    "Doctor: Let's check your vitals."
                )
            },
            "model_output": {
                "output": (
                    "• Chief complaint: Persistent cough and fever for three days\n"
                    "• History of present illness: The patient reports fatigue and occasional shortness of breath.\n"
                    "• Physical examination: Performed; temperature elevated at 38.5°C.\n"
                    "• Symptoms experienced by the patient: Cough, fever, fatigue, shortness of breath.\n"
                    "• New medications prescribed or changed: Prescribed antibiotics (Amoxicillin 500mg three times daily).\n"
                    "• Follow-up instructions: Return if symptoms worsen.\n"
                    "• Name: John Doe\n"  # Included disallowed term to induce failure
                )
            },
            # Expected outcomes:
            # - LLM Assertions: One PASS, one FAIL
            # - Code Assertions: One PASS, one FAIL
        },
        # Example 3
        {
            "task_description": (
                "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy "
                "guidelines and specified formatting instructions."
            ),
            "input_data": {
                "dialogue": (
                    "Doctor: Tell me about your knee pain.\n"
                    "Patient: It started after I twisted it playing soccer last weekend.\n"
                    "Doctor: On a scale of 1 to 10, how bad is the pain?\n"
                    "Patient: It's around a 6, worse when I move it.\n"
                    "Doctor: I'll examine it and see what's going on."
                )
            },
            "model_output": {
                "output": (
                    "• Chief complaint: Knee pain after twisting injury\n"
                    "• History of present illness: Patient reports pain level of 6 out of 10, exacerbated by movement.\n"
                    "• Physical examination: Swelling observed; limited range of motion.\n"
                    "• Symptoms experienced by the patient: Pain, swelling, reduced mobility.\n"
                    "• New medications prescribed or changed: N/A.\n"
                    "• Follow-up instructions: Rest, apply ice, and avoid strenuous activities.\n"
                    # Included 'Age' to induce failure
                    "• Age: 30\n"
                )
            },
            # Expected outcomes:
            # - LLM Assertions: One PASS, one FAIL
            # - Code Assertions: One PASS, one FAIL
        },
    ]

    # Initialize the AssertionScorer with the assertions
    scorer = AssertionScorer(
        assertions=assertions,
        llm_model="gpt-4o-2024-08-06",
        prompt_template="""
Task Description:
{task_description}

Evaluate the following output based on the given task, input, and assertion:

Input:
{input_data}

Output:
{model_output}

Assertion:
{assertion_text}

Consider the task description and input when evaluating the output against the assertion.
Respond with either 'PASS' if the output meets the assertion criteria in the context of the task and input, or 'FAIL' if it does not.
""",
        system_prompt="You are an AI assistant evaluating the quality of text outputs based on given tasks, inputs, and assertions.",
    )

    evaluation = weave.Evaluation(
        scorers=[scorer],
        dataset=examples,
    )

    await evaluation.evaluate(predict_passthrough)


if __name__ == "__main__":
    asyncio.run(main())
