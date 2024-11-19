from typing import List, Literal, Union

from pydantic import BaseModel, Field


class TaskDescription(BaseModel):
    description: str = Field(
        ..., description="A concise yet comprehensive task description"
    )


class CombinedTaskDescription(BaseModel):
    description: str = Field(
        ...,
        description="A comprehensive task description that combines LLM and human insights",
    )


class Criterion(BaseModel):
    criterion: str = Field(
        ...,
        description="A concise, specific statement describing a single aspect of evaluation",
    )
    explanation: str = Field(
        "",
        description="A detailed explanation of the criterion's importance and potential evaluation methods",
    )
    evaluation_method: Literal["code", "llm"] = Field(
        "",
        description="The primary method for evaluating this criterion: 'code' for programmatic checks, 'llm' for language model-based assessment",
    )

    def __hash__(self):
        return hash((self.criterion, self.explanation, self.evaluation_method))

    def __eq__(self, other):
        if isinstance(other, Criterion):
            return (self.criterion, self.explanation, self.evaluation_method) == (
                other.criterion,
                other.explanation,
                other.evaluation_method,
            )
        return False


class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="A list of 1-2 distinct evaluation criteria, each focusing on a different aspect of output quality",
    )


class PythonAssertion(BaseModel):
    test_name: str = Field(
        ...,
        description="A clear, concise name for the unittest.TestCase method with function name in snake case and starting with test_. Must be identical to the function name in the code field.",
    )
    code: str = Field(
        ...,
        description="A clear, concise assertion written as a unittest.TestCase method with function name in snake case and starting with test_. Also ensure that the input to the function is only self and the function accesses self.output.",
    )
    evaluation_type: Literal["python"] = "python"


class LLMAssertion(BaseModel):
    test_name: str = Field(
        ..., description="A clear, concise name for the LLM evaluator in snake case."
    )
    text: str = Field(
        ...,
        description="A detailed prompt for the LLM to evaluate the assertion, guiding it to return 'PASS' or 'FAIL'",
    )
    evaluation_type: Literal["llm"] = "llm"


class CriterionAssertions(BaseModel):
    assertions: List[Union[PythonAssertion, LLMAssertion]] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Generate 1-3 specific, testable assertions that can be used to evaluate LLM outputs based on the given criterion",
    )


class AssertionEvaluation(BaseModel):
    result: Literal["PASS", "FAIL"] = Field(
        ...,
        description="The evaluation result of an assertion. Must be either 'PASS' or 'FAIL'.",
    )


class DatasetMapping(BaseModel):
    input_data: str = Field(
        ...,
        description="The key in the sample data that corresponds to the input_data field in the DataPoint object",
    )
    output_data: str = Field(
        ...,
        description="The key in the sample data that corresponds to the output_data field in the DataPoint object",
    )
    annotation: str = Field(
        ...,
        description="The key in the sample data that corresponds to the annotation field in the DataPoint object",
    )
    note: str = Field(
        ...,
        description="The key in the sample data that corresponds to the note field in the DataPoint object",
    )
