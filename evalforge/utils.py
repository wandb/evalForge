import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time
import functools
from typing import Callable, Any, Optional
import asyncio
import pprint as pp
from contextlib import contextmanager

# Add global console instance
console = Console()


def pprint(d, indent=4, width=100):
    """Pretty print a dictionary or other object with line width control.

    Args:
        d: Dictionary or object to print
        indent: Number of spaces for indentation
        width: Maximum line width before wrapping
    """
    # For all objects, use the built-in pprint with width control
    printer = pp.PrettyPrinter(indent=indent, width=width)
    printer.pprint(d)


def load_jsonl(filename: Path | str) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with open(filename, "r") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data: list[dict], filename: Path | str):
    """Save a list of dictionaries to a JSONL file."""

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return obj

    with open(filename, "w") as file:
        for example in data:
            json.dump(example, file, default=convert_numpy)
            file.write("\n")


def listify(listable: list[str]) -> str:
    """Creates a markdown list of the items in the list."""
    if not listable:
        return "- None"
    return "\n".join([f"- {item}" for item in listable])


def sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Safely process messages for LiteLLM by converting all content to plain strings.
    This prevents issues with class attributes and non-pickleable objects.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    Returns:
        List of sanitized message dictionaries
    """
    return [
        {"role": str(msg["role"]), "content": str(msg["content"])} for msg in messages
    ]


async def tqdm_gather(coros, desc: Optional[str] = None, total: Optional[int] = None):
    """Create a Rich progress bar for gathering multiple coroutines

    Args:
        coros: List of coroutines to execute concurrently
        desc: Description for the progress bar
        total: Total number of steps (defaults to len(coros) if not provided)

    Returns:
        List of results from the gathered coroutines
    """
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    if total is None:
        total = len(coros)

    task_id = progress.add_task(f"[bold blue]{desc}", total=total)

    async def wrapped_coro(coro):
        result = await coro
        progress.update(task_id, advance=1)
        return result

    progress.start()
    try:
        results = await asyncio.gather(*[wrapped_coro(coro) for coro in coros])
        return results
    finally:
        progress.stop()


# Keep the original tqdm for synchronous operations
def tqdm(iterable=None, desc: str = None, total: int = None):
    """Create a Rich progress bar for synchronous operations"""
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    # Use provided total or calculate from coroutines
    if total is None:
        total = len(iterable)

    task_id = progress.add_task(f"[bold blue]{desc}", total=total)
    progress.start()
    try:
        for item in iterable:
            yield item
            progress.update(task_id, advance=1)
    finally:
        progress.stop()


def timer(func: Callable) -> Callable:
    """Decorator that measures and prints execution time of functions.
    Works with both async and regular functions.

    Args:
        func: The function to be timed

    Returns:
        Wrapped function that prints its execution time
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        console.print(f"[dim](time: {elapsed:.2f}s)[/]")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        console.print(f"[dim](time: {elapsed:.2f}s)[/]")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class Logger:
    def __init__(self):
        self.console = Console()

    def rule(self, name: str, color: str = "green") -> None:
        self.console.rule(f"[bold {color}]Begin {name}")

    def info(self, message: str):
        """Print an info message"""
        self.console.print(f"[green]► {message}[/]")

    def warning(self, message: str):
        """Print a warning message"""
        self.console.print(f"[yellow]► {message}[/]")

    def error(self, message: str):
        """Print an error message"""
        self.console.print(f"[red]► {message}[/]")

    def header(self, message: str):
        """Print a header message"""
        self.console.print(f"[bold blue]{message}[/]")

    @contextmanager
    def timer(self, message: str = None):
        """Context manager for timing operations

        Args:
            message: Optional message to print before timing
        """
        if message:
            self.info(message)

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.console.print(f"[dim](time: {elapsed:.2f}s)[/]")


# Create global logger instance
logger = Logger()
