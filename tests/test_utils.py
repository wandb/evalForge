import json
from pathlib import Path
import pytest
import numpy as np
from pydantic import BaseModel

from evalforge.utils import (
    pprint,
    load_jsonl,
    save_jsonl,
    listify,
    SuperEncoder,
)


# Test data
class TestModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def temp_jsonl(tmp_path):
    file_path = tmp_path / "test.jsonl"
    test_data = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    with open(file_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    return file_path, test_data


def test_load_jsonl(temp_jsonl):
    file_path, expected_data = temp_jsonl
    loaded_data = load_jsonl(file_path)
    assert loaded_data == expected_data


def test_save_jsonl(tmp_path):
    file_path = tmp_path / "output.jsonl"
    test_data = [
        {"normal": "data"},
        {"numpy": np.int64(42)},
        {"array": np.array([1, 2, 3])},
        {"pydantic": TestModel(name="test", value=1)},
    ]

    save_jsonl(test_data, file_path)
    loaded_data = load_jsonl(file_path)

    assert loaded_data[0] == {"normal": "data"}
    assert loaded_data[1] == {"numpy": 42}
    assert loaded_data[2] == {"array": [1, 2, 3]}
    assert loaded_data[3] == {"pydantic": {"name": "test", "value": 1}}


def test_listify():
    # Test empty list
    assert listify([]) == "- None"

    # Test normal list
    items = ["apple", "banana", "orange"]
    expected = "- apple\n- banana\n- orange"
    assert listify(items) == expected


def test_super_encoder():
    encoder = SuperEncoder()

    # Test numpy types
    assert pytest.approx(encoder.default(np.int64(42))) == 42
    assert pytest.approx(encoder.default(np.float32(3.14))) == 3.14
    assert encoder.default(np.bool_(True)) == True
    assert pytest.approx(encoder.default(np.array([1, 2, 3]))) == [1, 2, 3]

    # Test pydantic model
    model = TestModel(name="test", value=1)
    assert encoder.default(model) == {"name": "test", "value": 1}

    # Test unsupported type
    with pytest.raises(TypeError):
        encoder.default(set())


def test_pprint(capsys):
    # Test dictionary printing
    test_dict = {"a": 1, "b": 2}
    pprint(test_dict)
    captured = capsys.readouterr()
    assert eval(captured.out.strip()) == test_dict

    # Test string printing
    test_str = "Hello, World!"
    pprint(test_str)
    captured = capsys.readouterr()
    assert captured.out.strip() == f"'{test_str}'"
