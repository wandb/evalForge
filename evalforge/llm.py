from litellm import completion, acompletion
import instructor
from functools import wraps
import asyncio

from evalforge.utils import sanitize_messages


# we need this to fix litellm+weave bug
def sanitize_completion(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = sanitize_messages(kwargs["messages"])
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = sanitize_messages(kwargs["messages"])
        return func(*args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


llm_client = instructor.from_litellm(sanitize_completion(completion))
llm_aclient = instructor.from_litellm(sanitize_completion(acompletion))

# Default model configurations
DEFAULT_LLM_MODEL = "gpt-4o"  # For high accuracy tasks
DEFAULT_FAST_MODEL = "gpt-4o-mini"  # For faster, lighter tasks
