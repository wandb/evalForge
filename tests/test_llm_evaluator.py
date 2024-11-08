import pytest
from evalforge.llm_evaluator import LLMAssertionScorer
from evalforge.instructor_models import LLMAssertion


@pytest.fixture
def assertions():
    return [
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
    ]


@pytest.fixture
def task_description():
    return "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy guidelines and specified formatting instructions."


@pytest.fixture
def input_data():
    return {
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


@pytest.fixture
def output():
    return {
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


@pytest.mark.asyncio
async def test_llm_assertion_scorer(
    assertions, task_description, input_data, output
):
    scorer = LLMAssertionScorer(assertions=assertions)
    results = await scorer.score(
        output=output,
        input_data=input_data,
        task_description=task_description,
    )

    assert "llm_assertion_results" in results
    assert len(results["llm_assertion_results"]) == len(assertions)

    for test_name, score in results["llm_assertion_results"].items():
        assert isinstance(score, int)
        assert score in [0, 1]
