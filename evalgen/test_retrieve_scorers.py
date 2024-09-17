import weave
import openai

def setup_fake_llm_data():
    # Example task description, input data, and model output
    task_description = (
        "Transform a dialogue between a doctor and a patient into a structured medical note summary, adhering to privacy "
        "guidelines and specified formatting instructions."
    )
    
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
        "summary": (
            "• Chief complaint: Severe headaches for the past week\n"
            "• History of present illness: The patient reports daily headaches, particularly in the afternoon, accompanied "
            "by nausea and photophobia\n"
            "• Physical examination: Performed, details not provided\n"
            "• Symptoms experienced by the patient: Headaches, nausea, light sensitivity\n"
            "• New medications prescribed or changed: N/A\n"
            "• Follow-up instructions: N/A"
        )
    }

    return task_description, input_data, model_output

def test_retrieve_llm_assertion_scorer():
    task_description, input_data, model_output = setup_fake_llm_data()

    # Initialize OpenAI client
    client = openai.OpenAI()

    llm_assertion_scorer = weave.ref("weave:///a-sh0ts/llm_evaluator_test/object/LLMAssertionScorer:jkmOJ8eXadyjvHYd1De0g5D47h0MYm0lKhSrv9sUk7Q").get()
    print(llm_assertion_scorer)
    results = llm_assertion_scorer.score(model_output, task_description, input_data, client)

    print("Evaluation Results:")
    for test_name, result in results["llm_assertion_results"].items():
        print(f"{test_name}: {'PASS' if result == True else 'FAIL'}")

def test_retrieve_code_test_result_scorer():
    task_description, input_data, model_output = setup_fake_llm_data()

    code_test_result_scorer = weave.ref("weave:///a-sh0ts/llm_evaluator_test/object/CodeTestResultScorer:jkmOJ8eXadyjvHYd1De0g5D47h0MYm0lKhSrv9sUk7Q").get()
    print(code_test_result_scorer)

    # Score the model output
    scores = code_test_result_scorer.score(model_output, input_data, task_description)

    # Print the results
    print("Code Assertion Evaluation Results:")
    print(scores)

if __name__ == "__main__":
    test_retrieve_llm_assertion_scorer()
    # test_retrieve_code_test_result_scorer()