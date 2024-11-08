import pytest

from evalforge.forge import EvalForge
from evalforge.data_utils import load_data

@pytest.mark.vcr(
    filter_headers=["authorization", "x-api-key"],
    allowed_hosts=["api.openai.com", "localhost"],
)
@pytest.fixture
def datasets():
    DATASET_PATH = "data/mini_data.jsonl"
    raw_dataset = load_data(DATASET_PATH)

    train_ds = raw_dataset[:5]
    eval_ds = raw_dataset[5:]
    return {"train": train_ds, "eval": eval_ds}

@pytest.mark.vcr(
    filter_headers=["authorization", "x-api-key"],
    allowed_hosts=["api.openai.com", "localhost"],
)
@pytest.mark.asyncio
async def test_forge_training(datasets):
    # Create an instance of EvalForge
    LLM_MODEL = "gpt-4o"
    forger = EvalForge(
        batch_size=1,
        num_criteria_to_generate=1,
        llm_model=LLM_MODEL
    )

    # Run the fit method
    results = await forger.fit(datasets["train"])
    forged_judge = results["forged_judges"]["judge"]

    # Assertions to verify outputs
    assert forged_judge is not None
    assert "finalized_task_description" in results
    assert results["finalized_task_description"] != ""
    return forged_judge