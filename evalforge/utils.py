import json
from pathlib import Path

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

def save_jsonl(data: list[dict], filename: Path | str):
    """Save a list of dictionaries to a JSONL file."""
    with open(filename, "w") as file:
        for example in data:
            json.dump(example, file)
            file.write("\n")

def listify(l: list[str]) -> str:
    """Creates a markdown list of the items in the list."""
    if not l:
        return "- None"
    return "\n".join([f"- {item}" for item in l])