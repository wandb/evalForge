
import asyncio
import random
import json
from pathlib import Path

from evalforge.utils import load_jsonl
from evalforge.forge import EvalForge

import weave
weave.init(f"evalforge_test_{random.randint(0, 1000000)}")

# from evalforge.forge import EvalForge, convert_datapoint_to_example
# from evalforge.alignment import calculate_alignment_metrics, format_alignment_metrics


data = load_jsonl("data/helpful_reviews_annotations_1000_o1_mini_good_bad.jsonl")

# tuplify data: DataPoint = Tuple[dict, dict, Literal[0, 1], Optional[str], Optional[str], Optional[str]]

data = [({"text": example["item"][""]}, 
         {"text": example["item"]["description"]}, 
         example["target"], 
         "") for example in train_data]

data = data[:10]

print(data[0])

print("=" * 80)
print("Forging judge...")
forger = EvalForge()
results = asyncio.run(forger.fit(data))


forged_judge = results["forged_judges"]["judge"]
forged_judges_metrics = results["forged_judges"]["alignment_metrics"]
forged_judges_assertion_results = results["forged_judges"]["assertion_results"]
forged_judges_summary = results["forged_judges"]["summary"]

raw_judges = results["raw_judges"]["judge"]
raw_judges_metrics = results["raw_judges"]["alignment_metrics"]
raw_judges_assertion_results = results["raw_judges"]["assertion_results"]
raw_judges_summary = results["raw_judges"]["summary"]

annotation_examples = results["annotation_examples"]
finalized_task_description = results["finalized_task_description"]

# print("=" * 80)
# print("Forged judges summary:")
# print(forged_judges_summary)
# print("=" * 80)
# print("Raw judges summary:")
# print(raw_judges_summary)
# print("=" * 80)
# print(finalized_task_description)


# @weave.op
# async def run_assertions_and_calculate_metrics(forger, judge, data, task_description):
#     all_annotation_examples = convert_datapoint_to_example(task_description, data)
#     all_data_forged_judge_assertion_results = await forger.run_assertions(judge, all_annotation_examples)
#     all_data_metrics = calculate_alignment_metrics(all_data_forged_judge_assertion_results)
#     all_data_metrics_str = format_alignment_metrics(all_data_metrics)
#     return all_data_metrics_str


# print("=" * 80)
# print("Running assertions and calculating metrics...")
# assertions_and_metrics = asyncio.run(run_assertions_and_calculate_metrics(forger, forged_judge, data, finalized_task_description))
# print(assertions_and_metrics)
