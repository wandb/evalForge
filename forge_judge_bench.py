import asyncio
import random
import json
from pathlib import Path
from datasets import load_dataset

from evalforge.utils import load_jsonl, pprint
from evalforge.forge import EvalForge
from evalforge.data_utils import DataPoint
from evalforge.alignment import calculate_alignment_metrics, format_alignment_metrics

import weave
weave.init("evalforge_test_judgebench")

# Define number of samples to use
NUM_SAMPLES = 10

# load JudgeBench dataset
data = load_dataset("ScalerLab/JudgeBench", split="gpt")


# A datapoint
# {
#     "pair_id": "81ec57f2-483f-515f-93ff-78a8910b2153",                      # unique identifier for response pair
#     "original_id": "10646",                                                 # original question id in the source dataset
#     "source": "mmlu-pro-computer science",                                  # source dataset for question
#     "question": "Consider an additive white Gaussian noise channel ...",    # question to which responses are generated
#     "response_model": "gpt-4o-2024-05-13",                                  # model used to generate the responses
#     "response_A": "To determine the capacity of an additive white ...",     # one response candidate
#     "response_B": "We are given an additive white Gaussian noise ...",      # another response candidate
#     "label": "B>A"                                                          # objective label indicating correctness
# }


splitted_ds = data.train_test_split(seed=42, test_size=0.2)
train_ds = splitted_ds["train"].select(range(NUM_SAMPLES))  # limit to NUM_SAMPLES
test_ds = splitted_ds["test"].select(range(NUM_SAMPLES))    # limit to NUM_SAMPLES

print("One example from train dataset:")
pprint(train_ds[0])
print("=" * 80)

def generate_annotations(example: dict) -> DataPoint:
    label = 1 if example["label"] == "A>B" else 0
    return DataPoint(
        input_data={"text": example["question"]}, 
        output_data={"text": example["response_A"]}, 
        annotation=label, 
        note="",
    )

train_ds_formatted = [generate_annotations(example) for example in train_ds]
test_ds_formatted = [generate_annotations(example) for example in test_ds]

print("=" * 80)
print("Forging judge...")
forger = EvalForge(batch_size=1, num_criteria_to_generate=1)  # Add parameters like in mini
results = asyncio.run(forger.fit(train_ds_formatted))


forged_judge = results["forged_judges"]["judge"]
forged_judges_metrics = results["forged_judges"]["alignment_metrics"]
forged_judges_assertion_results = results["forged_judges"]["assertion_results"]
forged_judges_summary = results["forged_judges"]["summary"]

raw_judges = results["raw_judges"]["judge"]
raw_judges_metrics = results["raw_judges"]["alignment_metrics"]
raw_judges_assertion_results = results["raw_judges"]["assertion_results"]
raw_judges_summary = results["raw_judges"]["summary"]

finalized_task_description = results["finalized_task_description"]

print("Finalized task description:")
print(finalized_task_description)
print("=" * 80)
print("Forged judges summary:")
print(forged_judges_summary)
print("=" * 80)
print("Raw judges summary:")
print(raw_judges_summary)
print("=" * 80)

@weave.op
async def run_assertions_and_calculate_metrics(forger, judge, data):
    all_data_forged_judge_assertion_results = await forger.run_assertions(judge, data)
    all_data_metrics = calculate_alignment_metrics(all_data_forged_judge_assertion_results)
    all_data_metrics_str = format_alignment_metrics(all_data_metrics)
    return all_data_metrics_str


print("Running assertions and calculating metrics...")
assertions_and_metrics = asyncio.run(run_assertions_and_calculate_metrics(
    forger, forged_judge, test_ds_formatted))  # use formatted test data
print(assertions_and_metrics)
