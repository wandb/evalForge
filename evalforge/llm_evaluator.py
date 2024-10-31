import asyncio
from typing import Any, Dict, List, Optional


import weave
import instructor
from pydantic import Field


from evalforge.instructor_models import LLMAssertion
from evalforge.llm import llm_aclient, DEFAULT_LARGE_MODEL

class LLMAssertionScorer(weave.Scorer):
    assertions: List[LLMAssertion]
    model: str = Field(default=DEFAULT_LARGE_MODEL)
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

    @weave.op()
    async def score(
        self,
        model_output: Optional[Dict[str, Any]],
        task_description: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if model_output is None:
            return {"error": "No model output provided"}

        async def process_assertion(assertion):
            prompt = self.prompt_template.format(
                task_description=task_description,
                input_data=input_data,
                model_output=model_output["output"],
                assertion_text=assertion.text,
            )

            result = await llm_aclient.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_model=str,
            )

            # Map the LLM response to a score and standardize the result
            if result == "PASS":
                score = 1
            elif result == "FAIL":
                score = 0
            else:
                # Handle unexpected responses
                score = 0  # Treat unexpected responses as failures
                result = "FAIL"  # Standardize the result text

            # Return a dictionary similar to code assertions
            return assertion.test_name, {
                "score": score,
                "result": result,
                "type": "llm",
            }

        # Create tasks for all assertions
        tasks = [process_assertion(assertion) for assertion in self.assertions]

        # Run all tasks concurrently and gather results
        assertion_results = await asyncio.gather(*tasks)
        results = dict(assertion_results)

        return {"llm_assertion_results": results}