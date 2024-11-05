import asyncio
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
from evalforge.llm import llm_aclient, DEFAULT_LLM_MODEL
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
from evalforge.data_utils import DataPoint
from evalforge.utils import tqdm, logger


class EvalForge(weave.Model):

    MODEL: str = DEFAULT_LLM_MODEL
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
    task_description: Optional[str] = None

    def shuffle_and_batch_data(self, data: List[DataPoint]) -> List[List[DataPoint]]:
        "Shuffle and batch the data into smaller lists of datapoints"
        shuffled_data = random.sample(data, len(data))
        return [shuffled_data[i:i+self.batch_size] for i in range(0, len(shuffled_data), self.batch_size)]

    @weave.op
    async def get_task_description(self, data: List[DataPoint]) -> str:
        batched_data = self.shuffle_and_batch_data(data)
        task_description = ""
        
        with tqdm("Refining task description", min(4, len(batched_data))) as progress:
            for i, batch in enumerate(batched_data):
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
                progress.update(progress.task_id, advance=1)

        return response.description

    @weave.op
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

    @weave.op
    async def process_criteria(
        self, data: List[DataPoint], all_criteria: str
    ) -> EvaluationCriteria:
        formatted_data = DataPoint.format_batch(data)
        
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

    @weave.op
    async def generate_criteria(
        self, data: List[DataPoint], finalized_task_description: str
    ) -> List[Criterion]:
        all_criteria = []
        
        with tqdm("Generating criteria", self.num_criteria_to_generate) as progress:
            for _ in range(self.num_criteria_to_generate):
                response = await self.process_criteria(data, all_criteria)
                all_criteria.extend(response.criteria)
                progress.update(progress.task_id, advance=1)

        return all_criteria

    @weave.op
    async def create_candidate_assertions(
        self, data: List[DataPoint], criterion: Criterion
    ) -> CriterionAssertions:
        formatted_data = DataPoint.format_batch(data)
        prompt = self.candidate_assertion_prompt.format(
            formatted_data_string=formatted_data,
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

    @weave.op
    async def generate_all_assertions(self, criteria, data: List[DataPoint]):
        async def process_criterion(criterion):
            candidate_assertions = await self.create_candidate_assertions(
                data, criterion
            )
            assertions = candidate_assertions.assertions
            return criterion, assertions

        tasks = [process_criterion(criterion) for criterion in criteria]
        results = await asyncio.gather(*tasks)

        criterion_assertion_map = CriterionAssertionMap.from_assertions(results)

        return criterion_assertion_map

    @weave.op
    async def run_assertions(
        self, 
        scorer: AssertionScorer, 
        data: List[DataPoint]
    ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        criterion_assertion_results = {}

        async def process_example(datapoint: DataPoint):
            result = await scorer.score(
                model_output=datapoint.output_data,
                input_data=datapoint.input_data,
            )
            return result, datapoint.annotation

        results = []
        with tqdm("Running assertions on examples", len(data)) as progress:
            for datapoint in data:
                result = await process_example(datapoint)
                results.append(result)
                progress.update(progress.task_id, advance=1)

        # Process results
        for criterion_results, human_annotation in results:
            for criterion, assertion_results in criterion_results.items():
                if criterion not in criterion_assertion_results:
                    criterion_assertion_results[criterion] = {}
                for assertion_name, score in assertion_results.items():
                    if assertion_name not in criterion_assertion_results[criterion]:
                        criterion_assertion_results[criterion][assertion_name] = []
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

    @weave.op
    async def calculate_judge_metrics(
        self, 
        initial_scorer: AssertionScorer,
        assertion_results: Dict[str, Dict[str, List[Tuple[int, int]]]], 
        all_assertions: CriterionAssertionMap,
        criteria: List[Criterion]
    ) -> Tuple[Dict, Dict]:
        # Check if assertion_results is empty or None
        if not assertion_results:
            logger.warning("No assertion results found")
            return {}, {}

        initial_metrics = calculate_alignment_metrics(assertion_results)
        # Add check after calculating initial metrics
        if not initial_metrics:
            logger.warning("No metrics calculated from assertion results")
            return {}, {}

        best_assertions = select_best_assertions(
            initial_metrics,
            assertion_results,
            num_assertions_per_criterion=self.num_assertions_per_criterion,
        )
        filtered_assertion_results = filter_assertion_results(
            assertion_results, best_assertions
        )
        filtered_metrics = calculate_alignment_metrics(filtered_assertion_results)
        # Add check for filtered metrics
        if not filtered_metrics:
            logger.warning("No filtered metrics calculated")
            return {}, {}

        best_criteria = select_best_criteria(
            filtered_metrics, self.alignment_threshold, self.num_criteria
        )
        filtered_criterion_assertion_map = self._filter_best_assertions(
            best_criteria, all_assertions, criteria
        )

        final_judge = AssertionScorer(
            name="final_judge",
            criterion_assertion_map=filtered_criterion_assertion_map,
            llm_model=self.MODEL,
        )

        forged_metrics_str = format_alignment_metrics(filtered_metrics)
        initial_metrics_str = format_alignment_metrics(initial_metrics)

        forged_judges = {
            "judge": final_judge,
            "alignment_metrics": filtered_metrics,
            "assertion_results": filtered_assertion_results,
            "summary": forged_metrics_str,
        }
        initial_judges = {
            "judge": initial_scorer,
            "alignment_metrics": initial_metrics,
            "assertion_results": assertion_results,
            "summary": initial_metrics_str,
        }

        return forged_judges, initial_judges

    @weave.op
    async def fit(self, train_data: List[DataPoint]) -> Dict[str, Any]:
        logger.header("Starting EvalForge creation pipeline")
        
        with logger.timer("Generating task description"):
            llm_task_description = await self.get_task_description(train_data)
        
        with logger.timer("Combining human and LLM descriptions"):
            self.task_description = await self.combine_human_and_llm_descriptions(
                train_data, llm_task_description
            )
        
        with logger.timer("Generating evaluation criteria"):
            criteria = await self.generate_criteria(train_data, self.task_description)
        
        with logger.timer("Generating assertions"):
            all_assertions = await self.generate_all_assertions(criteria, train_data)
        
        with logger.timer("Running assertions on examples"):
            initial_scorer = AssertionScorer(
                criterion_assertion_map=all_assertions,
                llm_model=self.MODEL,
                task_description=self.task_description,
            )
            assertion_results = await self.run_assertions(initial_scorer, train_data)
        
        with logger.timer("Processing results"):
            forged_judges, initial_judges = await self.calculate_judge_metrics(
                initial_scorer,
                assertion_results, 
                all_assertions, 
                criteria
            )
        
        logger.header("EvalForge pipeline completed ✨")
        
        return {
            "forged_judges": forged_judges,
            "raw_judges": initial_judges,
            "task_description": self.task_description,
        }
