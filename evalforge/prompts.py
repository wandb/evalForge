import textwrap

# Task-related prompts
TASK_PROMPT = textwrap.dedent(
    """
    Current task description: {{ task_description }}

    New datapoints:
    {% for sample in samples %}
    Sample {{ loop.index }}:
    {% for key, value in sample.input_data.items() %}
    Input {{ key }}: {{ value }}
    {% endfor %}
    {% for key, value in sample.output_data.items() %}
    Output {{ key }}: {{ value }} 
    {% endfor %}
    Annotation: {{ 'Correct' if sample.annotation == 1 else 'Incorrect' }}
    {% if sample.note %}Note: {{ sample.note }}{% endif %}
    {% if not loop.last %}

    {% endif %}
    {% endfor %}

    Based on these new datapoints and the current task description, provide an updated, more refined task description. If this is the first batch, create an initial task description. Focus on:
    1. The nature of the input and output data
    2. The specific information being extracted or transformed
    3. Any formatting or style requirements
    4. Evaluation criteria (based on the annotations and notes)

    Keep the description concise yet comprehensive."""
)

TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assistant designed to help refine task descriptions for a given dataset.
    """
)

COMBINED_TASK_PROMPT = textwrap.dedent(
    """
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

    Please provide the combined description in a single, well-structured paragraph."""
)

COMBINED_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assistant designed to help refine task descriptions for a given dataset given a LLM-generated task description and additional human-provided context.
    """
)

# Criteria-related prompts
CRITERIA_PROMPT = textwrap.dedent(
    """
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
)

CRITERIA_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assistant designed to create evaluation criteria for a given task.
    """
)

# Assertion-related prompts
CANDIDATE_ASSERTION_PROMPT = textwrap.dedent(
    """
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
    """
)

CANDIDATE_ASSERTION_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assistant designed to create testable assertions for a given task and criterion.
    """
)

# LLM Assertion Scorer prompts
LLMASSERTION_PROMPT_TEMPLATE = textwrap.dedent(
    """
Task Description:
{task_description}

Evaluate the following output based on the given task, input, and assertion:

Input:
{input_data}

Output:
{output}

Assertion:
{assertion_text}

Consider the task description and input when evaluating the output against the assertion.
Respond with either 'PASS' if the output meets the assertion criteria in the context of the task and input, or 'FAIL' if it does not.
"""
)

LLMASSERTION_SYSTEM_PROMPT = "You are an AI assistant evaluating the quality of text outputs based on given tasks, inputs, and assertions."
