# 🚀 EvalGen Project

This project allows you to create and run LLM judges based on annotated datasets using Weights & Biases (wandb) and Weave for tracking and tracing.

## 🛠️ Setup

1. Create a `.env` file in the project root with the following variables:

```
WANDB_EMAIL=your_wandb_email 
WANDB_API_KEY=your_wandb_api_key
OPENAI_API_KEY=your_openai_api_key
```

1. Install the required dependencies.

## 🏃‍♂️ Running the Annotation App

To start the annotation app, run:

```bash
python main.py
```

This will launch a web interface for annotating your dataset.

## 🧠 Creating an LLM Judge

To programmatically create an LLM judge from your wandb dataset annotations:

1. Open `forge_evaluation_judge.ipynb` in a Jupyter environment.
2. Run all cells in the notebook.

This will generate a judge like the one in `forged_judge`.

## 🔍 Running the Generated Judge

To load and run the generated judge:

1. Open `run_forged_judge.ipynb` in a Jupyter environment.
2. Run all cells in the notebook.

This will evaluate your dataset using the forged judge, with results fully tracked and traced using Weave.

## 📊 Key Components

- `main.py`: Annotation app
- `forge_evaluation_judge.ipynb`: Judge creation notebook
- `run_forged_judge.ipynb`: Judge execution notebook

All components are integrated with Weave for comprehensive tracking and tracing of your machine learning workflow.

Happy evaluating! 🎉