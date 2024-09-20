import ast
import importlib
import os
import textwrap
from datetime import datetime
from typing import Dict, Optional, Set

import autopep8
import isort
import weave
from autoflake import fix_code


class CodeFormatter(weave.Object):
    @weave.op()
    def lint_code(self, code: str) -> str:
        code = code.replace("\\n", "\n")
        tree = ast.parse(code)
        required_imports = self.get_required_imports(tree)
        import_statements = self.generate_import_statements(required_imports)
        code = import_statements + "\n\n" + code
        code = fix_code(
            code, remove_all_unused_imports=True, remove_unused_variables=True
        )
        code = isort.code(code)
        code = autopep8.fix_code(code, options={"aggressive": 2})
        return code

    def get_required_imports(self, tree: ast.AST) -> Set[tuple]:
        required_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if not self.is_builtin(node.id):
                    required_imports.add((None, node.id))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        required_imports.add((node.module, alias.name))
        return required_imports

    def is_builtin(self, name: str) -> bool:
        return name in dir(__builtins__)

    def generate_import_statements(self, required_imports: Set[tuple]) -> str:
        import_statements = []
        for module, name in required_imports:
            try:
                if module:
                    importlib.import_module(module)
                    import_statements.append(f"from {module} import {name}")
                else:
                    importlib.import_module(name)
                    import_statements.append(f"import {name}")
            except ImportError:
                pass
        return "\n".join(import_statements)

    @weave.op()
    def write_assertions_to_files(
        self, assertions: Dict[str, str], base_dir: Optional[str] = None
    ) -> str:
        if base_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f"generated_assertions_{timestamp}"
        test_dir = os.path.join(base_dir, "tests")
        os.makedirs(test_dir, exist_ok=True)

        for assertion in assertions:
            assertion_name = assertion.test_name
            assertion_code = assertion.code
            file_name = f"test_{assertion_name}.py"
            full_code = self.create_test_file_content(assertion_name, assertion_code)
            with open(os.path.join(test_dir, file_name), "w") as f:
                f.write(full_code)

        # Create __init__.py in the test directory
        open(os.path.join(test_dir, "__init__.py"), "w").close()

        # Create run_tests.py
        run_tests_content = self.get_run_tests_content()
        with open(os.path.join(base_dir, "run_tests.py"), "w") as f:
            f.write(run_tests_content)

        return base_dir

    @weave.op()
    def create_test_file_content(self, assertion_name: str, assertion_code: str) -> str:
        # Dedent the assertion code to remove any existing indentation
        dedented_assertion_code = textwrap.dedent(assertion_code).strip()
        # Re-indent the assertion code to match the class indentation (4 spaces)
        indented_assertion_code = textwrap.indent(dedented_assertion_code, "    ")
        return f"""
import unittest
from run_tests import OutputTestCase

class Test_{assertion_name}(OutputTestCase):
{indented_assertion_code}

if __name__ == '__main__':
    unittest.main()
"""

    @weave.op()
    def get_run_tests_content(self) -> str:
        return """
import unittest
import sys
import os
import json

class OutputTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hasattr(cls, 'output'):
            return
        if len(sys.argv) < 2:
            raise ValueError("No output provided")
        try:
            cls.output = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided")

def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for test_class in unittest.defaultTestLoader.discover('.', pattern='test_*.py'):
        for test in test_class:
            suite.addTest(test)
    return suite

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py '<json_string>'")
        sys.exit(1)

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)

    # Exit with non-zero status if there were failures
    if not unittest.TextTestRunner().run(load_tests(None, None, None)).wasSuccessful():
        sys.exit(1)
"""


def main():
    # Usage example:
    code_formatter = CodeFormatter()

    assertions = {
        "within_word_limit": """
        def test_within_word_limit(self):
            # Count words in output
            total_words = sum(len(str(value).split()) for value in self.output['output'].split('\\n'))
            self.assertLessEqual(total_words, 150, f"Output exceeds word limit with {total_words} words.")
        """,
        "essential_information_inclusion": """
        def test_essential_information_inclusion(self):
            # Check for the presence of essential keys
            essential_keys = ['chief complaint', 'history of present illness', 'physical examination', 'symptoms experienced by the patient', 'new medications prescribed or changed', 'follow-up instructions']
            output_text = self.output['output'].lower()
            for key in essential_keys:
                self.assertIn(key, output_text, f"Output is missing essential information: {key}.")
        """,
        "no_excessive_information": """
        def test_no_excessive_information(self):
            # Check for any mention of PII or excessive details
            disallowed_terms = ['name', 'age', 'gender', 'ID']
            output_text = self.output['output'].lower()
            for term in disallowed_terms:
                self.assertNotIn(term, output_text, f"Output contains disallowed information: {term}.")
        """,
    }

    # Write assertions to files
    temp_dir = code_formatter.write_assertions_to_files(assertions)
    print(f"Generated assertions and tests written to: {temp_dir}")


if __name__ == "__main__":
    main()
