import asyncio
import simple_parsing
from simple_parsing import Serializable
from dataclasses import dataclass, Field
import sys

from evalforge.forge import EvalForge
from evalforge.utils import logger
from evalforge.data_utils import load_data, DataPoint

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

@dataclass
class Args(Serializable):
    data: str = "mini"# "Path to training data"
    batch_size: int = 1 # "Batch size"
    num_criteria_to_generate: int = 1 # "Number of criteria to generate"
    llm_model: str = "gpt-4o" # "LLM model to use"

def forge():
    logger.rule("EvalForge CLI")
    try:
        args = simple_parsing.parse(Args)

        # Load the data

        if args.data == "mini":
            logger.info(f"Running dummy data")
            train_data = train_ds_formatted
        else:
            logger.info(f"Loading data from {args.data}")
            train_data = load_data(args.data)
        
        forger = EvalForge(
            batch_size=args.batch_size, 
            num_criteria_to_generate=args.num_criteria_to_generate, 
            llm_model=args.llm_model
        )
        # Run the fit method
        asyncio.run(forger.fit(train_data))
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    forge() 