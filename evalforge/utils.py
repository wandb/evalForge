import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel


def pprint(d, indent=4):
    """Pretty print a dictionary or other object."""
    if isinstance(d, dict):
        print(json.dumps(d, indent=indent))
    else:
        print(d)

def load_jsonl(filename: Path | str) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with open(filename, "r") as file:
        return [json.loads(line) for line in file]
    
class BaseModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)
    
class SuperEncoder(BaseModelEncoder, NumpyEncoder):
    pass


def save_jsonl(data: list[dict], filename: Path | str):
    """Save a list of dictionaries to a JSONL file."""
    with open(filename, "w") as file:
        for example in data:
            json.dump(example, file, cls=SuperEncoder)
            file.write("\n")

def listify(l: list[str]) -> str:
    """Creates a markdown list of the items in the list."""
    if not l:
        return "- None"
    return "\n".join([f"- {item}" for item in l])