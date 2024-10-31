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
