import asyncio
from typing import Any, Dict, List, Optional

import openai
import weave
from pydantic import Field

from evalforge.instructor_models import LLMAssertion


class LLMAssertionScorer(weave.Scorer):
    assertions: List[LLMAssertion]
    model: str = Field(default="gpt-4o-2024-08-06")
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

        # Initialize OpenAI client
        client = openai.AsyncOpenAI()

        async def process_assertion(assertion):
            prompt = self.prompt_template.format(
                task_description=task_description,
                input_data=input_data,
                model_output=model_output["output"],
                assertion_text=assertion.text,
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            result = response.choices[0].message.content.strip()

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


def main():
    # Example usage
    weave.init("llm_evaluator_test")
    # Example LLM assertions
    assertions = [
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
        # Add more assertions as needed...
    ]

    # Create the LLMAssertionScorer
    scorer = LLMAssertionScorer(assertions=assertions)

    # Example task description, input data, and model output
    task_description = "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy guidelines and specified formatting instructions."

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
            "• History of present illness: The patient reports daily headaches, particularly in the afternoon, accompanied "
            "by nausea and photophobia.\n"
            "• Physical examination: Performed; details not provided.\n"
            "• Symptoms experienced by the patient: Headaches, nausea, light sensitivity.\n"
            "• New medications prescribed or changed: N/A.\n"
            "• Follow-up instructions: N/A."
        )
    }

    # Since the score method is asynchronous, use an async function to run it
    async def run_scorer():
        results = await scorer.score(model_output, task_description, input_data)
        print("Evaluation Results:")
        for test_name, result in results["llm_assertion_results"].items():
            print(f"{test_name}: {result}")

    # Run the async function
    asyncio.run(run_scorer())


if __name__ == "__main__":
    main()
