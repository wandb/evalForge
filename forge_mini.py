
import asyncio
import random
import json
from pathlib import Path
from datasets import load_dataset

from evalforge.utils import load_jsonl
from evalforge.forge import EvalForge
from evalforge.data_utils import DataPoint
from evalforge.alignment import calculate_alignment_metrics, format_alignment_metrics

import weave
weave.init("evalforge_test_judgebench")


train_ds_formatted = [
    DataPoint(
        input_data={"text": "1+1="}, 
        output_data={"text": "2"}, 
        annotation=1, 
        note="Correct summation",
    ), 
    DataPoint(
        input_data={"text": "1+1="}, 
        output_data={"text": "3"}, 
        annotation=0, 
        note="Incorrect summation",
    ),
    DataPoint(
        input_data={"text": "What is the square root of 16?"}, 
        output_data={"text": "4"}, 
        annotation=1, 
        note="Correct square root",
    ),
]

eval_ds_formatted = [
    DataPoint(
        input_data={"text": "What is the square root of 16?"}, 
        output_data={"text": "4"}, 
        annotation=1, 
        note="Correct square root",
    ),
    DataPoint(
        input_data={"text": "What is the square root of 16?"}, 
        output_data={"text": "3"}, 
        annotation=0, 
        note="Incorrect square root",
    ),
]

print("=" * 80)
print("Forging judge...")
forger = EvalForge(batch_size=1, num_criteria_to_generate=1)
results = asyncio.run(forger.fit(train_ds_formatted))


forged_judge = results["forged_judges"]["judge"]
forged_judges_metrics = results["forged_judges"]["alignment_metrics"]
forged_judges_assertion_results = results["forged_judges"]["assertion_results"]
forged_judges_summary = results["forged_judges"]["summary"]

raw_judges = results["raw_judges"]["judge"]
raw_judges_metrics = results["raw_judges"]["alignment_metrics"]
raw_judges_assertion_results = results["raw_judges"]["assertion_results"]
raw_judges_summary = results["raw_judges"]["summary"]



print("=" * 80)
print("Forged judges summary:")
print(forged_judges_summary)
print("=" * 80)
print("Raw judges summary:")
print(raw_judges_summary)
print("=" * 80)

finalized_task_description = results["finalized_task_description"]

print("Finalized task description:")
print(finalized_task_description)
print("=" * 80)

@weave.op
async def run_assertions_and_calculate_metrics(forger, judge, data):
    all_data_forged_judge_assertion_results = await forger.run_assertions(judge, data)
    all_data_metrics = calculate_alignment_metrics(all_data_forged_judge_assertion_results)
    all_data_metrics_str = format_alignment_metrics(all_data_metrics)
    return all_data_metrics_str


print("=" * 80)
print("Running assertions and calculating metrics...")
assertions_and_metrics = asyncio.run(run_assertions_and_calculate_metrics(
    forger, forged_judge, eval_ds_formatted))
print(assertions_and_metrics)
