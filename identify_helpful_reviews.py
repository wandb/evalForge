import instructor
# from litellm import acompletion

import openai
import os 

from datetime import datetime
import aiofiles

import weave
from pathlib import Path
from evalforge.utils import *

from typing import Literal
from pydantic import BaseModel, Field

import logging

import asyncio
from tqdm.asyncio import tqdm

from simple_parsing import ArgumentParser
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HELPFUL_VOTE_THRESHOLD = 5

weave.init("capecape/amazon_fashion")

# llm_client = instructor.from_litellm(acompletion)
oai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
instructor_client = instructor.from_openai(oai_client)

# MODEL_NAME = "o1-mini"
MODEL_NAME= "gpt-4o-mini"

existing_keys = set()
# lock = asyncio.Lock()

output_file_path = "data/find_helpful_reviews_data.jsonl"

system_prompt = """# Constructing a LLM Judge Benchmark

## The Benchmark
I am trying to build a benchmark for an LLM judge. This benchmark requires positive and negative labels for a given AI-generated output. I have a dataset of Amazon product descriptions and customer reviews which I think I can use. I will construct a benchmark dataset of product descriptions and an associated good / bad ground truth set of labels for the quality of the description text.
I will then pass the product descriptions to a LLM Judge and ask it to rate the descriptions. Then I will compare the Judge's labels with my ground truth labels in order to understand how aligned.

## Using Reviews As A Proxy signal For Description Text Quality
I want to use the product descriptions as a proxy for AI-generated output and use the feedback provided in the customer review texts as a proxy signal into the quality of the description text.

## Product Description Constraints
However I do not have the actual products in my hand so I am constrained to only assessing the quality of the description text, without knowing if the text matches the actual product in reality. Therefore I am just assessing whether the description text is clear and well-written or whether it is missing information or badly formatted or uses bad english, bad grammar etc. I cannot know if the description text is misleading as I cannot compare to the actual product in reality.

## Scoring Rubric
Given the customer reviews of descriptions please judge the following

- `is_useful`: Is the review useful in identifying good or bad descriptions.

<useful_review_definition>
A useful review is one where the review text provides a clear signal about the quality of the product description. The sentiment of the review doesn't matter - it could be a positive or negative review.

For example a useful review could be one that indicates a bad description, for instance if the product description is missing key information or is badly worded, uses poor grammar or is poorly styled or formatted. On the other hand a useful review could indicate a good product description, for example if the reviewer finds that the product description is accurate and helpful. For a review to be useful for this task, the review should explicitly mention the product description in some way.
</useful_review_definition>

<not_useful_review_definition>
A review is not useful for this task if:
- it doesn't mention the product description
- it says the description is misleading, as we cannot know whether or not the description is misleading without also having the product in our hands to compare it to
- it only makes indirect or implicit references to the description being good or bad
- it requires experience using, handling, holding, wearing or otherwise interacting with the product to make a judgement about the description.
</not_useful_review_definition>

## Examples
{examples}
"""


examples = """
Examples of reviews that are useful and not useful:

### useful examples

- I ordered 3 of these shirts and am mostly satisfied. The shirts are great for exercising and the sizing is pretty accurate. My only complaint is that I think I'd rather a cotton/polyester blend instead of 100% polyester. While the polyester keeps you dry, the shirt is a little light and slinky and I'd rather some of the weight that cotton would bring. Other that that, these shirts are well made and are as described
- Hoop is just as described, fit my 7 year old perfectly. It did have a smell when we first opened it but I didn't notice it nearly as much once it aired out. My daughter loved what it did to her dress. Would love to post picts just don't know how!
- This purse was exactly as described and pictured. I was nervous since I have always carried huge shoulder bags so this was a down size for me. I have been wanting a cross body purse so I don't have to do everything with one hand It had enough room to support my kindle voyage, a diary, pens, bill fold, portable charger for phone, my huge keychain with my thousand of tags, phone, and chap stick. Wonderful and will be ordering from this seller again
- Great boots. I checked on UGG website and they are authentic. I believe they are true to size. I would advise against buying a size smaller, as suggested by the description. These are my first UGG boots. So far I am very happy with them. These are my first UGG boots. So far I am very happy with them
- Everything I hoped it would be (except the size)<br />Love the fabric on my skin, soft & lightweight as described
- I don't understand how this has so many stars from people, these sandals run extremely small. The title says men's 34, but should say unisex and have a better description for sizes.
- I ordered 2 of these shirts ant they are great, but it didn't say that is is 100% polyester.
- The watch didn't include the wristband, I would expect a full functioning watch!


### non useful examples
- I marked these as &#34;fit as expected&#34;, but not at first. This was the third attempt to get a fitting pair. Ecco is notorious for mislabeling the US equivalent size, and these were no exception
- I recieved an empty freakin G-shock box and knew it from the time i picked it up it was so under weight!!!!! I want my watch or a refund ASAP!
- I purchased 1 previously & luv it! I had  another plain anklet that just broke so I said, hey, why not have matching ones!!! It goes with everything & I luv the way it sparkles!! Looks expensive!
- So the small is for anyone who is an adult female size 5 or smaller.  Just FYI.  Medium is recommended for female US 5-9.  I would check out the socks on another site for size features before ordering to ensure you're getting the right size for your feet. I ordered blindly for a gift; fortunately, the person is small and the small socks fit her well, despite her shoe size being a US 7.
- The gold hoop earrings were a nice shiny gold and the shape and size were as described. However, it is very difficult to close the clasp because it is very thin and flimsy.. it feels that if I am not very careful, the loops will bend. Somewhat disappointed in this regard, but not surprised given the price. Also, someone else mentioned in their review that they were difficult to clasp.
- I purchased these bc they are supposed to be waterproof. Mine are not.  Wore them in Colorado in 36 degree weather & I can feel air coming into the toe bed so my toes were cold.  Maybe they didn’t waterproof mine all the way idk or not at all bc mine are not.

"""

prompt_template = """
The item to review is:

<review_title>
{review_title}
</review_title>

<review_text>
{review_text}
</review_text>

<instructions>
First consider whether the review requires the reviewer to interact with the product to make a judgement about the quality of the product description.
Then consider reason(s) for why the review is useful or not useful in assessing the quality of the product description.
If product interaction is required to make an assessment then the review is not useful.
Then score if this review `is_useful`. 
</instructions>

"""


class ReviewEvaluation(BaseModel):
    requires_product_interaction: bool = Field(description="Does the review require the reviewer to interact with the product to make a judgement about the quality of the product description?")
    thinking: str = Field(description="Reason(s) for why the review is useful or not useful in assessing the quality of the product description.\
if product interaction is required to make an assessment then the review is not useful.")
    is_useful: bool = Field(description="Is the review useful in identifying good or bad descriptions?")


def format_example(review: dict):
    return prompt_template.format(
        review_title=review["title"],
        review_text=review["text"],
    )


async def async_map(func, items, max_concurrent=5, desc="Processing"):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def wrapped_func(item):
        async with semaphore:
            return await func(**item)  # Unpack the dictionary into keyword arguments
    
    tasks = [wrapped_func(item) for item in items]
    return await tqdm.gather(*tasks, desc=desc)

@weave.op
async def extract_review_evaluation(review_evaluation: str) -> ReviewEvaluation:
    """Extract the review evaluation from the LLM response"""
    review_evaluation = await instructor_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": review_evaluation}
        ],
        response_model=ReviewEvaluation,
    )
    return review_evaluation.model_dump()

@weave.op
async def call_openai(prompt: str, max_completion_tokens: int = 8000) -> dict:
    if "o1" in MODEL_NAME:
        o1_prompt = system_prompt.format(examples=examples) + prompt # no system prompt for o1
        out = await oai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": o1_prompt}],
            max_completion_tokens=max_completion_tokens,
        )
    else:        
        out = await oai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt.format(examples=examples)},
                {"role": "user", "content": prompt}],
            max_completion_tokens=max_completion_tokens,
        )
    review_evaluation = await extract_review_evaluation(out.choices[0].message.content)
    # logger.info(f"Review evaluation: prompt: {prompt[:-10]}")
    return {"review_evaluation": review_evaluation}


def get_key(review: dict):
    return f"{review['asin']}_{review['user_id']}_{review['timestamp']}"


async def load_existing_keys(output_file_path: str):
    """Load existing review keys into the in-memory set."""
    global existing_keys
    if os.path.exists(output_file_path):
        async with aiofiles.open(output_file_path, mode="r") as f:
            async for line in f:
                if line.strip():
                    try:
                        existing_review = json.loads(line)
                        key = existing_review.get("review_key")
                        if key:
                            existing_keys.add(key)
                    except json.JSONDecodeError:
                        logger.error(f"Malformed line in {output_file_path}: {line}")
                        continue  # Skip malformed lines

@weave.op
async def evaluate_review(review: dict) -> dict:
    key = get_key(review)

    # async with lock:
    if key in existing_keys:
        logger.debug(f"Skipping review {key} as it already exists")
        res = {
            # "review": review,
            "review_evaluation": {
                "requires_product_interaction": False,
                "thinking": "Review already processed.",
                "is_useful": False
            },
            "skipped": True,
            "review_key": key
        }
    else:
        # Mark the key as processed to prevent duplicates in concurrent executions
        existing_keys.add(key)

        """Evaluate a single review using the LLM"""
        o1_prompt = format_example(review)
        out = await call_openai(o1_prompt)
        # res = {"review": review, **out, "skipped": False}
        res = {
            "review_evaluation": out["review_evaluation"],
            "skipped": False,
            "review_key": key
        }

        async with aiofiles.open(output_file_path, mode="a") as f:
            await f.write(f'{{"review_key": "{key}", "review": {json.dumps(review)}, "review_evaluation": {json.dumps(out["review_evaluation"])}}}\n')

    return res

@weave.op
async def evaluate_reviews(reviews: list[dict], max_concurrent: int = 25) -> list[dict]:
    combined_args = [{'review': review} for review in reviews]
    
    return await async_map(
        evaluate_review,
        combined_args,
        max_concurrent=max_concurrent,
        desc="Processing reviews"
    )


async def main(reviews: list[dict], max_concurrent: int):
    await load_existing_keys(output_file_path)
    tstamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    with weave.attributes({"helpful_threshold": HELPFUL_VOTE_THRESHOLD,
                        "n_reviews": len(reviews),
                        "run_tstamp": tstamp}):
        annotations = await evaluate_reviews(reviews, max_concurrent=max_concurrent)
        print(f"DONE! Number of annotations: {len(annotations)}")

@dataclass
class Args:
    start_index: int = 0
    end_index: int = 10
    max_concurrent: int = 25

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    dataset_path = Path("notebooks/amazon_clothing_and_jewelry_helpful.jsonl")
    reviews = load_jsonl(dataset_path)

    reviews = reviews[args.start_index:args.end_index]

    asyncio.run(main(reviews, max_concurrent=args.max_concurrent))