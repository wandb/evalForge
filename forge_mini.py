import asyncio

from evalforge.utils import logger
from evalforge.forge import EvalForge
from evalforge.data_utils import load_data
from evalforge.alignment import calculate_alignment_metrics, format_alignment_metrics

import weave

weave.init("evalforge_test_judgebench")


# train_ds_formatted = [
#     DataPoint(
#         input_data={"text": "1+1="},
#         output_data={"text": "2"},
#         annotation=1,
#         note="Correct summation",
#     ),
#     DataPoint(
#         input_data={"text": "1+1="},
#         output_data={"text": "3"},
#         annotation=0,
#         note="Incorrect summation",
#     ),
#     DataPoint(
#         input_data={"text": "What is the square root of 16?"},
#         output_data={"text": "4"},
#         annotation=1,
#         note="Correct square root",
#     ),
# ]

# eval_ds_formatted = [
#     DataPoint(
#         input_data={"text": "What is the square root of 16?"},
#         output_data={"text": "4"},
#         annotation=1,
#         note="Correct square root",
#     ),
#     DataPoint(
#         input_data={"text": "What is the square root of 16?"},
#         output_data={"text": "3"},
#         annotation=0,
#         note="Incorrect square root",
#     ),
# ]

LLM_MODEL = "gpt-4o"
DATASET_PATH = "data/mini_data.jsonl"
dataset = load_data(DATASET_PATH)
train_ds_formatted = dataset[:5]
eval_ds_formatted = dataset[5:]

forger = EvalForge(batch_size=1, num_criteria_to_generate=1, llm_model=LLM_MODEL)
results = asyncio.run(forger.fit(train_ds_formatted))
forged_judge = results["forged_judges"]["judge"]

logger.rule("Running assertions and calculating metrics", color="blue")


@weave.op
async def run_assertions_and_calculate_metrics(forger, judge, data):
    all_data_forged_judge_assertion_results = await forger.run_assertions(judge, data)
    all_data_metrics = calculate_alignment_metrics(
        all_data_forged_judge_assertion_results
    )
    format_alignment_metrics(all_data_metrics)
    return


asyncio.run(
    run_assertions_and_calculate_metrics(forger, forged_judge, eval_ds_formatted)
)
