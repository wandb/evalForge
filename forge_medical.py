import asyncio
import random
from pathlib import Path

import weave

from evalforge.utils import logger, pprint
from evalforge.forge import EvalForge
from evalforge.data_utils import load_data, DataPoint
from evalforge.combined_scorer import AssertionScorer
from evalforge.alignment import calculate_alignment_metrics, format_alignment_metrics

weave.init("evalforge_test_judgebench")

ds_formatted = weave.ref(
    "weave:///a-sh0ts/medical_data_results/object/medical_data_annotations:7GcCtWgyPTWtKY48Z7v5VxwCNZXTTTpSMbmubAbyHT8"
).get()
data = random.sample(ds_formatted, 10)

# def to_datapoint(d):
#     p = dict(
#         input_data = {"test": d[0]["input"]},
#         output_data = {"test": d[1]["output"]},
#         annotation = int(d[2]),
#         note = d[3]
#     )
#     return DataPoint.model_validate(p)

# formatted_data = [to_datapoint(d) for d in data]

formatted_data = load_data(data)

LLM_MODEL = "gpt-4o"
BATCH_SIZE = 1
NUM_CRITERIA_TO_GENERATE = 3

forger = EvalForge(
    batch_size=BATCH_SIZE,
    num_criteria_to_generate=NUM_CRITERIA_TO_GENERATE,
    llm_model=LLM_MODEL,
)
results = asyncio.run(forger.fit(formatted_data))
forged_judge = results["forged_judges"]["judge"]

logger.info(f"Forged judge: {forged_judge.model_dump()}")

logger.rule("Running assertions and calculating metrics", color="blue")


@weave.op
async def run_assertions_and_calculate_metrics(
    forger: EvalForge, judge: AssertionScorer, data: list[DataPoint]
):
    all_data_forged_judge_assertion_results = await forger.run_assertions(judge, data)
    all_data_metrics = calculate_alignment_metrics(
        all_data_forged_judge_assertion_results
    )
    format_alignment_metrics(all_data_metrics)
    return


asyncio.run(run_assertions_and_calculate_metrics(forger, forged_judge, formatted_data))
