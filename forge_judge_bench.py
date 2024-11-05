
import asyncio
import random
import json
from pathlib import Path
from datasets import load_dataset

from evalforge.utils import load_jsonl
from evalforge.forge import EvalForge
from evalforge.data_utils import DataPoint

import weave
weave.init("evalforge_test_judgebench")

# load JudgeBench dataset
data = load_dataset("ScalerLab/JudgeBench", split="gpt")


# A sample datapoint:
# {'pair_id': 'e302b0a0-28d5-5a3c-b1af-fedcf5543e72',
#  'original_id': 1420,
#  'source': 'mmlu-pro-law',
#  'question': "A college student initiated a criminal case against his former roommate, claiming the former roommate snuck in and removed his state-of-the-art home theater equipment from his house. The college student took the complaint papers that were approved by the prosecution and filed them with a magistrate court. An ongoing police investigation later determined, however, that the college student was a vengeful person not averse to creating stories, and that he and the former roommate had been arguing over several related issues. They also determined that he never had any quality sound or video equipment. The police dropped the prosecution, and the criminal case was dismissed at the preliminary hearing. When the former roommate filed a civil case against the college student alleging the tort of malicious prosecution, will evidence of the student's reputation for dishonesty be admissible and why?\n(A) Yes, because the student's reputation for dishonesty directly relates to the tort of malicious prosecution.\n(B) No, because reputation evidence is considered too prejudicial to be admissible.\n(C) Yes, because dishonesty at any time in the past is always relevant to a malicious prosecution complaint.\n(D) Yes, because the student's dishonesty directly affects the credibility of his initial criminal complaint.\n(E) Yes, because reputation evidence, just like most character evidence, is virtually always admissible.\n(F) Yes, because even though character evidence is too collateral to be admitted for circumstantial evidence, it is admissible if it is directly at issue in the case.\n(G) No, because character evidence is only admissible if it is directly related to the act in question.\n(H) No, because the past reputation of a person is not considered as evidence in court.\n(I) Yes, because while most character evidence is not admissible, some is, and reputation evidence is always admissible.\n(J) No, because character evidence is never admissible in any case.\nIf you cannot determine the correct multiple-choice answer, take your best guess. Once you have your answer, please duplicate that letter five times in a single string. For example, if the answer is K, then write KKKKK.\nLet's think step by step.",
#  'response_model': 'gpt-4o-2024-05-13',
#  'response_A': 'To determine if evidence of the student\'s reputation for dishonesty is admissible in a claim of malicious prosecution, we need to consider several legal principles, especially those concerning character evidence and its relevance.\n\n1. **Relevance to Malicious Prosecution**:\n   - Malicious prosecution involves proving that the defendant initiated a legal action without probable cause and with malice. Evidence suggesting the initiator\'s dishonesty could help show the lack of probable cause and the element of malice.\n\n2. **General Rules on Character Evidence**:\n   - Generally, character evidence (reputation, opinion, or specific instances of conduct) is not admissible to prove conduct on a particular occasion (Federal Rules of Evidence 404(a)).\n   - There are exceptions where character evidence is admissible, typically in cases where character itself is an essential element of a claim or defense (e.g., defamation, negligent hiring, etc.).\n\n3. **Specific to Malicious Prosecution**:\n   - For malicious prosecution, the plaintiff would need to show that the college student acted with malice in initiating the criminal case.\n   - The student\'s reputation for dishonesty is relevant to understanding whether the student acted with malice or fabricated the entire situation.\n\nAnalyzing the options step-by-step:\n\n(A) Yes, because the student\'s reputation for dishonesty directly relates to the tort of malicious prosecution.\n   - This is a strong consideration since dishonesty could indicate malice and lack of probable cause.\n\n(B) No, because reputation evidence is considered too prejudicial to be admissible.\n   - While reputation evidence can be prejudicial, it is not an absolute bar, especially when the character is a pertinent element.\n\n(C) Yes, because dishonesty at any time in the past is always relevant to a malicious prosecution complaint.\n   - The phrase "always relevant" is too broad and may be misleading.\n\n(D) Yes, because the student\'s dishonesty directly affects the credibility of his initial criminal complaint.\n   - This is true, but it should focus on relevance to the malicious prosecution claim rather than just general credibility.\n\n(E) Yes, because reputation evidence, just like most character evidence, is virtually always admissible.\n   - Incorrect, as most character evidence rules say it is generally not admissible unless specific exceptions apply.\n\n(F) Yes, because even though character evidence is too collateral to be admitted for circumstantial evidence, it is admissible if it is directly at issue in the case.\n   - This correctly addresses that character is directly at issue in the case of malicious prosecution.\n\n(G) No, because character evidence is only admissible if it is directly related to the act in question.\n   - This is correct but incomplete in understanding how the student\'s character for dishonesty specifically relates to malice and the claim.\n\n(H) No, because the past reputation of a person is not considered as evidence in court.\n   - Incorrect, past reputation can be considered where it is directly related to the issues of the case.\n\n(I) Yes, because while most character evidence is not admissible, some is, and reputation evidence is always admissible.\n   - Incorrect, not always admissible.\n\n(J) No, because character evidence is never admissible in any case.\n   - Incorrect, as there are specific contexts in which it is admissible.\n\nThe best-balanced reasoning aligns with choice (F), as it clearly states that when character is directly at issue, such as in a malicious prosecution claim, it becomes admissible.\n\nHence my answer is:\nFFFFF',
#  'response_B': "To determine whether the former roommate can use evidence of the college student's reputation for dishonesty in his civil case for malicious prosecution, we need to consider rules of admissibility of character evidence, especially reputation evidence, in the context of a civil lawsuit.\n\n1. **Understand the nature of the civil lawsuit**:\n   - The case is about malicious prosecution, which involves proving that the defendant (college student) initiated an unjust legal proceeding with malice and without probable cause.\n\n2. **Evaluate the rules of evidence for character/reputation**:\n   - In legal proceedings, character evidence is generally inadmissible to prove that a person acted in accordance with that character on a specific occasion.\n   - However, character evidence can be admissible if it is directly related to proving an element of a claim, defense, or charge.\n\n3. **Relevance to malicious prosecution**:\n   - Malicious prosecution claims often focus on the intent and credibility of the defendant.\n   - The student's reputation for dishonesty might be directly relevant to showing that the initial complaint was maliciously false.\n\n4. **Admissibility based on relevance**:\n   - Federal Rules of Evidence (FRE) and many state rules allow the introduction of character evidence if the trait is pertinent to the issue at hand. In this case, dishonesty is a core aspect of the malicious prosecution claim.\n   - The character trait of dishonesty would be directly relevant because it pertains to the potential fabrication of the initial complaint.\n\nBased on these considerations, the best answer is:\n(A) Yes, because the student's reputation for dishonesty directly relates to the tort of malicious prosecution.\n\nSo, the correct and best-supported answer is:\nAAAAA",
#  'label': 'A>B'}


splitted_ds = data.train_test_split(seed=42, test_size=0.2)
train_ds = splitted_ds["train"]
test_ds = splitted_ds["test"]

def generate_annotations(example: dict) -> DataPoint:
    label = 1 if example["label"] == "A>B" else 0
    return DataPoint(
        input_data={"text": example["question"]}, 
        output_data={"text": example["response_A"]}, 
        annotation=label, 
        note="",
    )

train_ds_formatted = [generate_annotations(example) for example in train_ds][:8]

print("=" * 80)
print("Forging judge...")
forger = EvalForge()
results = asyncio.run(forger.fit(train_ds_formatted))


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
