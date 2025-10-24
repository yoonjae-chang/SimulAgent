import os

from dotenv import load_dotenv

from dedalus_labs import Dedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_sync

from autogluon.tabular import TabularPredictor
import pandas as pd
import tempfile
import os

load_dotenv()

from autogluon.tabular import TabularPredictor
import pandas as pd

def answer_question_with_autogluon(csv_path: str, label_column: str, decision_column: str, option_1: str, option_2: str, time_limit: int = 60) -> str:
    """
    Trains a model and recommends the better option between option_1 and option_2 for decision_column.
    """
    print("Running AutoGluon model training...")

    # Train model
    predictor = TabularPredictor(label=label_column).fit(csv_path, time_limit=time_limit)
    
    # Load the dataset to compute expected outcomes
    df = pd.read_csv(csv_path)
    
    # Filter by options
    df_opt1 = df[df[decision_column] == option_1]
    df_opt2 = df[df[decision_column] == option_2]

    # Get predicted outcomes
    pred_opt1 = predictor.predict(df_opt1)
    pred_opt2 = predictor.predict(df_opt2)

    # Compare mean predictions
    mean1 = pred_opt1.mean()
    mean2 = pred_opt2.mean()

    # Generate recommendation
    if mean1 > mean2:
        recommendation = f"Release **{option_1}** — it yields higher predicted `{label_column}` (${mean1:.2f} vs ${mean2:.2f})."
    else:
        recommendation = f"Release **{option_2}** — it yields higher predicted `{label_column}` (${mean2:.2f} vs ${mean1:.2f})."

    return recommendation

def autogluon_tool(csv_path: str, label_column: str, decision_column: str, option_1: str, option_2: str, time_limit: int = 60) -> str:
    """
    Tool: autogluon_tool
    Purpose: Select the better option using a trained AutoGluon model.
    Inputs: csv_path (str), label_column (str), decision_column (str), option_1 (str), option_2 (str)
    Output: String with recommendation
    """
    print("🛠️ AutoGluon tool was called with:")
    print(f"  csv_path={csv_path}")
    print(f"  label_column={label_column}")
    print(f"  decision_column={decision_column}")
    print(f"  option_1={option_1}")
    print(f"  option_2={option_2}")
    return answer_question_with_autogluon(csv_path, label_column, decision_column, option_1, option_2, time_limit)


def main():
    client = Dedalus()
    runner = DedalusRunner(client)

    result = runner.run(
        input="""
Use the `autogluon_tool` to answer this question:

csv_path: 'data/products.csv'
label_column: 'revenue'
decision_column: 'product_type'
option_1: 'YC slideshow'
option_2: 'Seed round slideshow'

Question: Which digital slideshow should I release to make the most money?

Use AutoGluon to train a model and return the better option.
State if you use the tool and explain how.
""",
        model=["openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514"],
        tools=[autogluon_tool],
        mcp_servers=[
            "windsor/brave-search-mcp",
            "kaggle-z9ezws-31"
        ],
        stream=True,
    )
    stream_sync(result)
    print()

if __name__ == "__main__":
    main()
