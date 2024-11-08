import asyncio
from typing import Optional
import simple_parsing
from simple_parsing import Serializable
from dataclasses import dataclass
import sys

import weave

from evalforge.forge import EvalForge
from evalforge.utils import logger
from evalforge.data_utils import load_data

MINI_DATASET_PATH = "data/mini_data.jsonl"


@dataclass
class Args(Serializable):
    data: str = "mini"  # "Path to training data"
    batch_size: int = 1  # "Batch size"
    num_criteria_to_generate: int = 1  # "Number of criteria to generate"
    llm_model: str = "gpt-4o"  # "LLM model to use"
    weave_project: Optional[str] = None  # "Weave project to use"


def forge():
    logger.rule("EvalForge CLI")
    try:
        args = simple_parsing.parse(Args)

        # Log into Weave
        if args.weave_project:
            weave.init(args.weave_project)

        # Load the data
        if args.data == "mini":
            logger.info("Running dummy data")
            train_data = load_data(MINI_DATASET_PATH)
        else:
            logger.info(f"Loading data from {args.data}")
            train_data = load_data(args.data)

        forger = EvalForge(
            batch_size=args.batch_size,
            num_criteria_to_generate=args.num_criteria_to_generate,
            llm_model=args.llm_model,
        )
        # Run the fit method
        asyncio.run(forger.fit(train_data))
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    forge()
