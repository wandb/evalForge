import pytest
import ast
import os
import textwrap
from evalforge.code_formatter import CodeFormatter
from evalforge.instructor_models import PythonAssertion


@pytest.fixture
def code_formatter():
    return CodeFormatter()


@pytest.fixture
def sample_assertions():
    return {
        "within_word_limit": textwrap.dedent(
            """
            def test_within_word_limit(self):
                # Count words in output
                total_words = sum(len(str(value).split()) for value in self.output['output'].split('\\n'))
                self.assertLessEqual(total_words, 150, f"Output exceeds word limit with {total_words} words.")
        """
        ).strip(),
        "essential_information_inclusion": textwrap.dedent(
            """
            def test_essential_information_inclusion(self):
                # Check for the presence of essential keys
                essential_keys = ['chief complaint', 'history of present illness', 'physical examination']
                output_text = self.output['output'].lower()
                for key in essential_keys:
                    self.assertIn(key, output_text, f"Output is missing essential information: {key}.")
        """
        ).strip(),
    }


def test_lint_code(code_formatter):
    sample_code = textwrap.dedent(
        """
        def test_function(self):
            output_text = self.output['output']
            self.assertIsInstance(output_text,    str)
            self.assertTrue(output_text.strip( ))
    """
    ).strip()
    formatted_code = code_formatter.lint_code(sample_code)
    # Verify no syntax errors in formatted code
    ast.parse(formatted_code)
    # Check basic formatting
    assert "strip()" in formatted_code  # removed extra spaces


def test_write_assertions_to_files(code_formatter, sample_assertions, tmp_path):
    # Write assertions to files
    base_dir = code_formatter.write_assertions_to_files(
        [
            PythonAssertion(test_name=name, code=code, evaluation_type="python")
            for name, code in sample_assertions.items()
        ],
        base_dir=str(tmp_path),
    )

    # Check if test files are created with correct content
    for assertion_name in sample_assertions:
        test_file = os.path.join(base_dir, "tests", f"test_{assertion_name}.py")
        assert os.path.exists(test_file)
        with open(test_file, "r") as f:
            content = f.read()
            assert f"class Test_{assertion_name}(OutputTestCase):" in content
