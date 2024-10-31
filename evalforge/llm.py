from litellm import completion, acompletion
import instructor

import weave # patches everything

llm_client = instructor.from_litellm(completion)
llm_aclient = instructor.from_litellm(acompletion)

# Default model configurations
DEFAULT_LARGE_MODEL = "gpt-4o"  # For high accuracy tasks
DEFAULT_FAST_MODEL = "gpt-4o-mini"  # For faster, lighter tasks
