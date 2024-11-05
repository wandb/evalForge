import asyncio
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from jinja2 import Template
import random

import weave
from litellm import acompletion

from evalforge.combined_scorer import AssertionScorer
from evalforge.criterion_assertion_map import CriterionAssertionMap
from evalforge.alignment import (calculate_alignment_metrics,
                                           filter_assertion_results,
                                           format_alignment_metrics,
                                           select_best_assertions,
                                           select_best_criteria)
from evalforge.instructor_models import (CombinedTaskDescription, Criterion,
                                         CriterionAssertions,
                                         EvaluationCriteria, TaskDescription)
from evalforge.llm import llm_aclient, DEFAULT_LARGE_MODEL
from evalforge.prompts import (
    TASK_PROMPT,
    TASK_SYSTEM_PROMPT,
    COMBINED_TASK_PROMPT,
    COMBINED_TASK_SYSTEM_PROMPT,
    CRITERIA_PROMPT,
    CRITERIA_SYSTEM_PROMPT,
    CANDIDATE_ASSERTION_PROMPT,
    CANDIDATE_ASSERTION_SYSTEM_PROMPT,
)
from evalforge.data_utils import (
    DataPoint,
    format_all_datapoints,
    convert_datapoint_to_example
)


class EvalForge(weave.Model):

    MODEL: str = DEFAULT_LARGE_MODEL
    task_prompt: str = TASK_PROMPT
    task_system_prompt: str = TASK_SYSTEM_PROMPT
    combined_task_prompt: str = COMBINED_TASK_PROMPT
    combined_task_system_prompt: str = COMBINED_TASK_SYSTEM_PROMPT
    num_criteria_to_generate: int = 3
    criteria_prompt: str = CRITERIA_PROMPT
    criteria_system_prompt: str = CRITERIA_SYSTEM_PROMPT
    candidate_assertion_prompt: str = CANDIDATE_ASSERTION_PROMPT
    candidate_assertion_system_prompt: str = CANDIDATE_ASSERTION_SYSTEM_PROMPT
    num_assertions_per_criterion: Optional[int] = None
    alignment_threshold: float = 0.4
    num_criteria: int = 3
    batch_size: int = 4

    def shuffle_and_batch_data(self, data: List[DataPoint]) -> List[List[DataPoint]]:
        "Shuffle and batch the data into smaller lists of datapoints"
        shuffled_data = random.sample(data, len(data))
        return [shuffled_data[i:i+self.batch_size] for i in range(0, len(shuffled_data), self.batch_size)]

    @weave.op()
    async def get_task_description(self, data: List[DataPoint]) -> str:
        batched_data = self.shuffle_and_batch_data(data)
        task_description = ""
        
        for i, batch in tqdm(enumerate(batched_data), desc="Refining task description"):
            if i > 3:
                break
            # Convert DataPoints to dictionaries for the template
            samples = [
                {
                    'input_data': dp.input_data,
                    'output_data': dp.output_data,
                    'annotation': dp.annotation,
                    'note': dp.note
                }
                for dp in batch
            ]
            
            template = Template(self.task_prompt)
            formatted_prompt = template.render(
                task_description=task_description,
                samples=samples
            )

            response = await llm_aclient.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.task_system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                response_model=TaskDescription
            )
            task_description = response.description

        return response.description

    @weave.op()
    async def combine_human_and_llm_descriptions(
        self, data: List[DataPoint], llm_description: str
    ) -> str:
        human_descriptions = set()
        for dp in data:
            if dp.human_description:  # Check if human description exists
                human_descriptions.add(dp.human_description)

        if not human_descriptions:
            return llm_description

        human_context = "\n".join(f"- {desc}" for desc in human_descriptions)

        prompt = self.combined_task_prompt.format(
            llm_description=llm_description, human_context=human_context
        )

        response = await llm_aclient.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.combined_task_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_model=CombinedTaskDescription,
        )

        return response.description

    @weave.op()
    async def process_criteria(
        self, formatted_data: str, all_criteria: str
    ) -> EvaluationCriteria:
        prompt = self.criteria_prompt.format(
            formatted_data=formatted_data,
            generated_criteria=str([c.model_dump() for c in all_criteria]),
        )
        
        response = await llm_aclient.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.criteria_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_model=EvaluationCriteria,
        )
        return response

    @weave.op()
    async def generate_criteria(
        self, data: List[DataPoint], finalized_task_description: str
    ) -> List[Criterion]:
        all_criteria = []
        formatted_data = format_all_datapoints(data, finalized_task_description)

        for _ in range(self.num_criteria_to_generate):
            response = await self.process_criteria(formatted_data, all_criteria)
            all_criteria.extend(response.criteria)

        return all_criteria

    @weave.op()
    async def create_candidate_assertions(
        self, formatted_data_string: str, criterion: Criterion
    ) -> CriterionAssertions:
        prompt = self.candidate_assertion_prompt.format(
            formatted_data_string=formatted_data_string,
            criterion=criterion.model_dump(),
        )
        response = await llm_aclient.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.candidate_assertion_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_model=CriterionAssertions,
        )
        return response

    @weave.op()
    async def generate_all_assertions(self, criteria, formatted_data):
        async def process_criterion(criterion):
            candidate_assertions = await self.create_candidate_assertions(
                formatted_data, criterion
            )
            assertions = candidate_assertions.assertions
            return criterion, assertions

        tasks = [process_criterion(criterion) for criterion in criteria]
        results = await asyncio.gather(*tasks)

        # Use the alternative constructor
        criterion_assertion_map = CriterionAssertionMap.from_assertions(results)

        return criterion_assertion_map

    @weave.op()
    async def run_assertions(
        self, scorer: AssertionScorer, annotation_examples: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        # The outer dict maps criterion names to assertion results
        criterion_assertion_results = {}

        async def process_example(example):
            result = await scorer.score(
                model_output={"output": example["model_output"]["output"]},
                task_description=example["task_description"],
                input_data=example["input_data"],
            )
            return result, example["annotation"]

        # Run examples one by one
        results = []
        for example in annotation_examples:
            result = await process_example(example)
            results.append(result)

        # # Run all examples concurrently
        # results = await asyncio.gather(
        #     *[process_example(example) for example in annotation_examples]
        # )
        # Process the results to accumulate scores
        for criterion_results, human_annotation in results:
            # criterion_results is a dict mapping criteria to their assertion results
            for criterion, assertion_results in criterion_results.items():
                if criterion not in criterion_assertion_results:
                    criterion_assertion_results[criterion] = {}
                for assertion_name, score in assertion_results.items():
                    if assertion_name not in criterion_assertion_results[criterion]:
                        criterion_assertion_results[criterion][assertion_name] = []
                    # Append the (score, human_annotation) tuple
                    criterion_assertion_results[criterion][assertion_name].append(
                        (score, human_annotation)
                    )

        return criterion_assertion_results

    def _filter_best_assertions(self, best_criteria, all_assertions, criteria):
        """Private helper method to filter assertions based on best criteria"""
        filtered_criterion_assertion_map = CriterionAssertionMap()
        original_criteria = {c.criterion: c for c in criteria}

        for criterion_name, criterion_data in best_criteria.items():
            if criterion_name in original_criteria:
                original_criterion = original_criteria[criterion_name]
                best_assertion_names = set(criterion_data["per_assertion"].keys())

                assertions = all_assertions.get_assertions_by_criterion(criterion_name)
                if assertions:
                    for assertion in assertions:
                        if assertion.test_name in best_assertion_names:
                            filtered_criterion_assertion_map.add_assertion(
                                original_criterion, assertion
                            )

        return filtered_criterion_assertion_map

    @weave.op()
    async def predict(self, data: List[DataPoint]) -> List[float]:
        llm_task_description = self.get_task_description(data)
        finalized_task_description = await self.combine_human_and_llm_descriptions(
            data, llm_task_description
        )
        criteria = await self.generate_criteria(data, finalized_task_description)
        formatted_data = format_all_datapoints(data, finalized_task_description)
        all_assertions = await self.generate_all_assertions(criteria, formatted_data)
        annotation_examples = convert_datapoint_to_example(
            finalized_task_description, data
        )
        scorer = AssertionScorer(
            criterion_assertion_map=all_assertions,
            llm_model=self.MODEL,
        )
        assertion_results = await self.run_assertions(scorer, annotation_examples)
        metrics = calculate_alignment_metrics(assertion_results)
        best_assertions = select_best_assertions(
            metrics,
            assertion_results,
            num_assertions_per_criterion=self.num_assertions_per_criterion,  # Use intelligent selection
        )
        filtered_assertion_results = filter_assertion_results(
            assertion_results, best_assertions
        )
        new_metrics = calculate_alignment_metrics(filtered_assertion_results)
        best_criteria = select_best_criteria(
            new_metrics, self.alignment_threshold, self.num_criteria
        )
        filtered_criterion_assertion_map = self._filter_best_assertions(
            best_criteria, all_assertions, criteria
        )

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
