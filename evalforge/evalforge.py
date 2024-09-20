import openai
import json
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import instructor
import weave
from set_env import set_env
from pydantic import BaseModel, Field
from typing import List, Literal, Union, Any
from pprint import pprint
from evalforge.instructor_models import TaskDescription, CombinedTaskDescription, Criterion, EvaluationCriteria, PythonAssertion, LLMAssertion, CriterionAssertions
import asyncio
import nest_asyncio
from evalforge.criterion_assertion_map import CriterionAssertionMap
from evalforge.combined_scorer import AssertionScorer, predict_passthrough
from evalforge.evalforge_alignment import calculate_alignment_metrics, select_best_assertions, filter_assertion_results, select_best_criteria, format_alignment_metrics

client = instructor.from_openai(openai.AsyncOpenAI())
DataPoint = Tuple[dict, dict, Literal[0, 1], Optional[str], Optional[str], Optional[str]]  # (input, output, annotation, note, human_description_for_task_or_judge, human_description_for_metric_details)

def format_single_datapoint(dp: DataPoint, finalized_task_description: str) -> str:
    input_data, output_data, annotation, note = dp[0], dp[1], dp[2], dp[3]
    metrics_details = dp[5] if len(dp) > 5 else None

    formatted = [
        f"Task Description: {finalized_task_description}",
        "",
        "Input:",
        "\n".join(f"  {key.capitalize()}: {value}" for key, value in input_data.items()),
        "",
        "Output:",
        "\n".join(f"  {key.capitalize()}: {value}" for key, value in output_data.items()),
        "",
        f"Annotation: {'Correct' if annotation == 1 else 'Incorrect'}",
        f"Note: {note}"
    ]

    if metrics_details:
        formatted.append(f"Metrics Details: {metrics_details}")

    return "\n".join(formatted)

#TODO: improve this function
def format_all_datapoints(data: List[DataPoint], finalized_task_description: str) -> str:
    formatted = [f"Task Description: {finalized_task_description}\n"]
    
    for i, dp in enumerate(data, 1):
        input_data, output_data, annotation, note = dp[0], dp[1], dp[2], dp[3]
        
        formatted.extend([
            f"Example {i}:",
            "Input:",
            json.dumps(input_data, indent=2),
            "",
            "Output:",
            json.dumps(output_data, indent=2),
            "",
            f"Annotation: {'Correct' if annotation == 1 else 'Incorrect'}",
            f"Note: {note}",
            "\n" + "-"*50 + "\n"  # Separator between examples
        ])
    
    return "\n".join(formatted)

def convert_datapoint_to_example(task_description: str, data: List[DataPoint]) -> List[Dict[str, Any]]:
    examples = []
    for dp in data:
        input_data, output_data, annotation, note = dp[0], dp[1], dp[2], dp[3]
        examples.append({
            "task_description": task_description,
            "input_data": input_data,
            "model_output": {"output": output_data},
            "annotation": annotation,
            "note": note
        })
    return examples

def filter_best_assertions(best_criteria, all_assertions, criteria):
    filtered_criterion_assertion_map = CriterionAssertionMap()
    original_criteria = {c.criterion: c for c in criteria}

    for criterion_name, criterion_data in best_criteria.items():
        if criterion_name in original_criteria:
            original_criterion = original_criteria[criterion_name]
            best_assertion_names = set(criterion_data['per_assertion'].keys())
            
            assertions = all_assertions.get_assertions_by_criterion(criterion_name)
            if assertions:
                for assertion in assertions:
                    if assertion.test_name in best_assertion_names:
                        filtered_criterion_assertion_map.add_assertion(
                            original_criterion,
                            assertion
                        )
    
    return filtered_criterion_assertion_map

class EvalForge(weave.Model):

    MODEL: str = "gpt-4o-2024-08-06"
    task_prompt: str = """
Current task description: {task_description}

New datapoint:
Input: {input_data}
Output: {output_data}
Annotation: {annotation}
Note: {note}

Based on this new datapoint and the current task description, provide an updated, more refined task description. 
If this is the first datapoint, create an initial task description.
Focus on:
1. The nature of the input and output data
2. The specific information being extracted or transformed
3. Any formatting or style requirements
4. Evaluation criteria (based on the annotation and note)

Keep the description concise yet comprehensive.
"""

    task_system_prompt: str = """
You are an AI assistant designed to help refine task descriptions for a given dataset.
"""
    combined_task_prompt: str = """
LLM-generated task description:
{llm_description}

Additional human-provided context:
{human_context}

Your task is to create a comprehensive, coherent task description that combines insights from both the LLM-generated description and the human-provided context. Ensure that:
1. The final description is clear and concise.
2. It incorporates key points from both sources.
3. Any contradictions are resolved logically.
4. The description maintains a professional tone.
5. It provides a complete picture of the task requirements and evaluation criteria.

Please provide the combined description in a single, well-structured paragraph.
"""
    combined_task_system_prompt: str = """
You are an AI assistant designed to help refine task descriptions for a given dataset given a LLM-generated task description and additional human-provided context.
"""
    num_criteria_to_generate: int = 3
    criteria_prompt: str = """
Analyze the following annotated datapoints:

{formatted_data}

Note we have already generated criteria, so we can use that as context:
{generated_criteria}

Generate 1 evaluation criteria that can be used to assess the quality of outputs for this task. Consider the following guidelines:

1. If a 'Metrics Details' field is present in the datapoint, prioritize this information as it provides the most important evaluation criteria.
2. Focus on general aspects of quality that can be used across multiple outputs.
3. Consider criteria that address potential misalignment between LLM outputs and human preferences.
4. Include criteria that can be evaluated both by code and by LLM-based evaluators.
5. Think about criteria that might reveal hallucinations, instruction-following, or other common LLM issues.
6. Generate criteria that could help in debugging or improving the LLM pipeline.

Provide the criterion as a concise statement, followed by a brief explanation of why it's important and how it might be evaluated (e.g., via code, LLM evaluator, or human judgment).

Return the criteria in this format:
[Criterion]: [Brief explanation and evaluation method]

Aim for a mix of straightforward, code-evaluable criteria and more nuanced criteria that might require LLM or human evaluation.
"""
    criteria_system_prompt: str = """
You are an AI assistant designed to create evaluation criteria for a given task.
"""
    candidate_assertion_prompt: str = """
Given the following evaluation criterion and annotated data, generate 1-3 specific, testable assertions:

Criterion: {criterion}

Annotated data: {formatted_data_string}

Your task is to create assertions that can be used to evaluate LLM outputs based on this criterion. Follow these guidelines:

1. Make each assertion clear, concise, and directly related to the criterion
2. For Python assertions:
- Provide a valid Python method that can be used within a unittest.TestCase class
- Ensure the method name is in snake case and starts with test_
- The method should take 'self' as the only input, where 'self.output' is a dictionary containing the LLM output being evaluated
- The 'self.output' dictionary will have the same keys and shape as the output in the annotated data
- Use unittest assertion methods (e.g., self.assertTrue, self.assertEqual) to test the output
- The test should pass if the assertion is met, and fail otherwise
- Only use the keys and shapes present in the annotated data output for your assertions
3. For LLM assertions:
- Provide a clear, detailed prompt for an LLM to evaluate the assertion
- The prompt should guide the LLM to return "PASS" or "FAIL" based on the evaluation
4. Include a mix of positive and negative assertions where appropriate
5. Consider edge cases and potential failure modes for the criterion
6. Aim for assertions that could be applied across multiple types of outputs

Ensure that your assertions are directly evaluable and avoid vague or subjective language. Focus on creating assertions that align with human preferences and can be used to validate the quality of LLM-generated evaluations.

Format your response as a JSON object with the following structure:
{{
"assertions": [
    {{
    "test_name": "Name of the test case method in snake case",
    "text" or "code": "Assertion text or code",
    "evaluation_type": "python" or "llm"
    }},
    ...
]
}}
"""
    candidate_assertion_system_prompt: str = """
You are an AI assistant designed to create testable assertions for a given task and criterion.
"""
    num_assertions_per_criterion: Optional[int] = None
    alignment_threshold: float = 0.4
    num_criteria: int = 3



    # TODO: Batch this as opposed to one at a time
    # or sample the dataset and ensure that taking into tokens (maybe something fun with a distribution)
    # distribution = more stuff we can grab and throw into prompt in smart way
    @weave.op()
    async def get_task_description(self, data: List[DataPoint]) -> str:
        task_description = ""
        
        for i, datapoint in enumerate(data):
            input_data, output_data, annotation, note = datapoint[0], datapoint[1], datapoint[2], datapoint[3]
            
            prompt = self.task_prompt.format(task_description=task_description, input_data=input_data, output_data=output_data, annotation="Correct" if annotation == 1 else "Incorrect", note=note)

            response = await client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.task_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_model=TaskDescription
            )
            
            new_description = response.description
            
            # TODO: Add guardrails to prevent LLM from saying no update needed
            if new_description.lower().startswith("no update needed"):
                continue
            
            task_description = new_description

        return task_description
    
    @weave.op()
    async def combine_human_and_llm_descriptions(self,data: List[DataPoint], llm_description: str) -> str:
        human_descriptions = set()
        for dp in data:
            if len(dp) > 4 and dp[4]:  # Check if human description exists
                human_descriptions.add(dp[4])
        
        if not human_descriptions:
            return llm_description
        
        human_context = "\n".join(f"- {desc}" for desc in human_descriptions)
        
        prompt = self.combined_task_prompt.format(llm_description=llm_description, human_context=human_context)

        response = await client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.combined_task_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_model=CombinedTaskDescription,
        )
        
        return response.description
    
    @weave.op()
    async def process_criteria(self, formatted_data: str, all_criteria: str) -> EvaluationCriteria:
        prompt = self.criteria_prompt.format(formatted_data=formatted_data, generated_criteria=str([c.model_dump() for c in all_criteria]))
        response = await client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.criteria_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_model=EvaluationCriteria
        )
        return response

    @weave.op()
    async def generate_criteria(self, data: List[DataPoint], finalized_task_description: str) -> List[Criterion]:
        all_criteria = []
        formatted_data = format_all_datapoints(data, finalized_task_description)

        for _ in range(self.num_criteria_to_generate):
            response = await self.process_criteria(formatted_data, all_criteria)
            all_criteria.extend(response.criteria)

        return all_criteria
    
    @weave.op()
    async def create_candidate_assertions(self, formatted_data_string: str, criterion: Criterion) -> CriterionAssertions:
        prompt = self.candidate_assertion_prompt.format(formatted_data_string=formatted_data_string, criterion=criterion.model_dump())
        response = await client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": self.candidate_assertion_system_prompt}, {"role": "user", "content": prompt}],
            response_model=CriterionAssertions
        )
        return response

    @weave.op()
    async def generate_all_assertions(self, criteria, formatted_data):
        async def process_criterion(criterion):
            candidate_assertions = await self.create_candidate_assertions(formatted_data, criterion)
            assertions = candidate_assertions.assertions
            return criterion, assertions

        tasks = [process_criterion(criterion) for criterion in criteria]
        results = await asyncio.gather(*tasks)

        # Use the alternative constructor
        criterion_assertion_map = CriterionAssertionMap.from_assertions(results)

        return criterion_assertion_map
    
    @weave.op()
    async def run_assertions(self, scorer: AssertionScorer, annotation_examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        # The outer dict maps criterion names to assertion results
        criterion_assertion_results = {}

        async def process_example(example):
            result = await scorer.score(
                model_output={"output": example["model_output"]["output"]},
                task_description=example["task_description"],
                input_data=example["input_data"]
            )
            return result, example["annotation"]

        # Run all examples concurrently
        results = await asyncio.gather(*[process_example(example) for example in annotation_examples])

        # Process the results to accumulate scores
        for (criterion_results, human_annotation) in results:
            # criterion_results is a dict mapping criteria to their assertion results
            for criterion, assertion_results in criterion_results.items():
                if criterion not in criterion_assertion_results:
                    criterion_assertion_results[criterion] = {}
                for assertion_name, score in assertion_results.items():
                    if assertion_name not in criterion_assertion_results[criterion]:
                        criterion_assertion_results[criterion][assertion_name] = []
                    # Append the (score, human_annotation) tuple
                    criterion_assertion_results[criterion][assertion_name].append((score, human_annotation))

        return criterion_assertion_results
    
    @weave.op()
    async def predict(self, data: List[DataPoint]) -> List[float]:
        llm_task_description = await self.get_task_description(data)
        finalized_task_description = await self.combine_human_and_llm_descriptions(data, llm_task_description)
        criteria = await self.generate_criteria(data, finalized_task_description)
        formatted_data = format_all_datapoints(data, finalized_task_description)
        all_assertions = await self.generate_all_assertions(criteria, formatted_data)
        annotation_examples = convert_datapoint_to_example(finalized_task_description, data)
        scorer = AssertionScorer(
            criterion_assertion_map=all_assertions,
            llm_model=self.MODEL,
        )
        assertion_results = asyncio.run(self.run_assertions(scorer, annotation_examples))
        metrics = calculate_alignment_metrics(assertion_results)
        best_assertions = select_best_assertions(
            metrics,
            assertion_results,
            num_assertions_per_criterion=self.num_assertions_per_criterion  # Use intelligent selection
        )
        filtered_assertion_results = filter_assertion_results(assertion_results, best_assertions)
        new_metrics = calculate_alignment_metrics(filtered_assertion_results)
        best_criteria = select_best_criteria(new_metrics, self.alignment_threshold, self.num_criteria)
        filtered_criterion_assertion_map = filter_best_assertions(best_criteria, all_assertions, criteria)

        final_judge = AssertionScorer(
            name="final_judge",
            criterion_assertion_map=filtered_criterion_assertion_map,
            llm_model=self.MODEL,
        )

        forged_alignment_metrics_str = format_alignment_metrics(new_metrics)

        raw_alignment_metrics_str = format_alignment_metrics(metrics)

        return {
            "forged_judges": {
                "judge": final_judge,
                "alignment_metrics": new_metrics,
                "assertion_results": filtered_assertion_results,
                "summary": forged_alignment_metrics_str,
            },
            "raw_judges": {
                "judge": scorer,
                "alignment_metrics": metrics,
                "assertion_results": assertion_results,
                "summary": raw_alignment_metrics_str,
            },
            "annotation_examples": annotation_examples,
            "finalized_task_description": finalized_task_description,
        }