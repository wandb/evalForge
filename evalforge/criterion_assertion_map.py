from typing import Dict, List, Union
from instructor_models import Criterion, LLMAssertion, PythonAssertion
from pydantic import BaseModel
import weave

class CriterionAssertionMap(weave.Object):
    criterion_to_assertions: Dict[str, List[Union[LLMAssertion, PythonAssertion]]] = {}
    assertion_to_criterion: Dict[str, str] = {}

    def add_assertion(self, criterion: Criterion, assertion: Union[LLMAssertion, PythonAssertion]):
        criterion_name = criterion.criterion
        if criterion_name not in self.criterion_to_assertions:
            self.criterion_to_assertions[criterion_name] = []
        self.criterion_to_assertions[criterion_name].append(assertion)
        self.assertion_to_criterion[assertion.test_name] = criterion_name

    def get_assertions_by_criterion(self, criterion_name: str) -> List[Union[LLMAssertion, PythonAssertion]]:
        return self.criterion_to_assertions.get(criterion_name, [])

    def get_criterion_by_assertion(self, assertion_name: str) -> str:
        return self.assertion_to_criterion.get(assertion_name)
    
    @classmethod
    def from_assertions(cls, criterion_assertion_pairs):
        instance = cls()
        for criterion, assertions in criterion_assertion_pairs:
            for assertion in assertions:
                instance.add_assertion(criterion, assertion)
        return instance