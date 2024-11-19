import asyncio
from typing import Any, Dict, List, Tuple

import weave
from pydantic import Field

from evalforge.instructor_models import LLMAssertion, AssertionEvaluation
from evalforge.llm import llm_aclient
from evalforge.prompts import LLMASSERTION_PROMPT_TEMPLATE, LLMASSERTION_SYSTEM_PROMPT


class LLMAssertionScorer(weave.Scorer):
    assertions: List[LLMAssertion] = Field(default_factory=list)
    model: str = Field(default="gpt-4")
    prompt_template: str = Field(default=LLMASSERTION_PROMPT_TEMPLATE)
    system_prompt: str = Field(default=LLMASSERTION_SYSTEM_PROMPT)

    async def process_assertion(
        self,
        assertion: LLMAssertion,
        *,  # Force kwargs
        output: Any,
        input_data: Any,
        task_description: str,
    ) -> Tuple[str, int]:
        formatted_prompt = self.prompt_template.format(
            task_description=task_description,
            input_data=input_data,
            output=output,
            assertion_text=assertion.text,
        )

        response = await llm_aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt},
            ],
            response_model=AssertionEvaluation,
        )

        score = 1 if response.result == "PASS" else 0
        return assertion.test_name, score

    @weave.op
    async def score(
        self,
        *,  # Force kwargs
        output: Any,
        input_data: Any,
        task_description: str,
    ) -> Dict[str, Dict[str, int]]:
        tasks = [
            self.process_assertion(
                assertion,
                output=output,
                input_data=input_data,
                task_description=task_description,
            )
            for assertion in self.assertions
        ]

        assertion_results = await asyncio.gather(*tasks)
        return {"llm_assertion_results": dict(assertion_results)}
