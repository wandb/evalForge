#!/bin/bash

echo "Running black formatting..."
black .

echo "Running isort import sorting..."
isort .

echo "Running ruff code linting and autofix..."
ruff check . --fix

echo "Running mypy type checking..."
mypy .

echo "Linting and formatting completed."