import json

def pprint(d, indent=4):
    """Pretty print a dictionary or other object."""
    if isinstance(d, dict):
        print(json.dumps(d, indent=indent))
    else:
        print(d)