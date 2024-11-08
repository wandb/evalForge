import pytest
from evalforge.data_utils import generate_mapping
from evalforge.instructor_models import DatasetMapping


@pytest.fixture
def list_sample():
    return [
        {"input": "What is the capital of France?"},
        {
            "output": "Paris is the capital of France.",
            "score": 1,
            "feedback": "Good answer, accurate and concise.",
        },
        1,
        "Good answer, accurate and concise.",
    ]


@pytest.fixture
def dict_sample():
    return {
        "question": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "score": 1,
        "feedback": "Good answer, accurate and concise.",
    }


@pytest.fixture
def nested_dict_sample():
    return {
        "data": {
            "input_text": "What is the capital of France?",
            "model_output": "Paris is the capital of France.",
            "evaluation": {
                "is_correct": 1,
                "reviewer_notes": "Good answer, accurate and concise.",
            },
        }
    }


def test_generate_mapping_list_structure(list_sample):
    mapping = generate_mapping(list_sample)
    assert isinstance(mapping, dict)
    assert "input_data" in mapping
    assert "output_data" in mapping
    assert "annotation" in mapping
    assert "note" in mapping

    # Test the mapping works with DataPoint.from_example
    from evalforge.data_utils import DataPoint

    data_point = DataPoint.from_example(list_sample, mapping)
    assert data_point is not None
    assert data_point.annotation in [0, 1]


def test_generate_mapping_dict_structure(dict_sample):
    mapping = generate_mapping(dict_sample)
    assert isinstance(mapping, dict)
    assert all(
        key in mapping for key in ["input_data", "output_data", "annotation", "note"]
    )

    # Verify the mapping matches expected structure
    from evalforge.data_utils import DataPoint

    data_point = DataPoint.from_example(dict_sample, mapping)
    assert data_point is not None
    assert data_point.annotation in [0, 1]


def test_generate_mapping_nested_structure(nested_dict_sample):
    mapping = generate_mapping(nested_dict_sample)
    assert isinstance(mapping, dict)
    assert all(
        key in mapping for key in ["input_data", "output_data", "annotation", "note"]
    )

    # Verify the mapping works with nested structures
    from evalforge.data_utils import DataPoint

    data_point = DataPoint.from_example(nested_dict_sample, mapping)
    assert data_point is not None
    assert data_point.annotation in [0, 1]


def test_generate_mapping_invalid_input():
    with pytest.raises(Exception):
        generate_mapping(None)
    with pytest.raises(Exception):
        generate_mapping([])
    with pytest.raises(Exception):
        generate_mapping({})


def test_mapping_model_validation():
    # Test that the mapping follows DatasetMapping model
    sample_mapping = {
        "input_data": "question",
        "output_data": "response",
        "annotation": "score",
        "note": "feedback",
    }
    mapping_model = DatasetMapping(**sample_mapping)
    assert mapping_model.model_dump() == sample_mapping
