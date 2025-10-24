import os
from dotenv import load_dotenv
from dedalus_labs import Dedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_sync
from autogluon.tabular import TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

load_dotenv()


# 🧠 AutoGluon Tool -------------------------------------------------------------

def answer_question_with_autogluon(csv_path: str, label_column: str, decision_column: str,
                                   option_1: str, option_2: str, time_limit: int = 60) -> str:
    print("Running AutoGluon model training...")

    predictor = TabularPredictor(label=label_column).fit(csv_path, time_limit=time_limit)

    df = pd.read_csv(csv_path)
    df_opt1 = df[df[decision_column] == option_1]
    df_opt2 = df[df[decision_column] == option_2]

    pred_opt1 = predictor.predict(df_opt1)
    pred_opt2 = predictor.predict(df_opt2)

    mean1 = pred_opt1.mean()
    mean2 = pred_opt2.mean()

    if mean1 > mean2:
        return f"Release **{option_1}** — higher predicted `{label_column}` (${mean1:.2f} vs ${mean2:.2f})."
    else:
        return f"Release **{option_2}** — higher predicted `{label_column}` (${mean2:.2f} vs ${mean1:.2f})."


def autogluon_tool(csv_path: str, label_column: str, decision_column: str,
                   option_1: str, option_2: str, time_limit: int = 60) -> str:
    print("🛠️ AutoGluon tool called with:")
    print(f"  csv_path={csv_path}")
    print(f"  label_column={label_column}")
    print(f"  decision_column={decision_column}")
    print(f"  option_1={option_1}")
    print(f"  option_2={option_2}")

    return answer_question_with_autogluon(csv_path, label_column, decision_column,
                                          option_1, option_2, time_limit)


# 📦 Kaggle Download Tool -------------------------------------------------------

def kaggle_download_tool(dataset_ref: str, download_dir: str = "./data") -> str:
    """
    Tool: kaggle_download_tool
    Purpose: Download and unzip a Kaggle dataset.
    Inputs:
        dataset_ref (str): Kaggle dataset ref, e.g. "zynicide/wine-reviews"
        download_dir (str): Directory to store dataset
    Output: Path to downloaded data
    """
    print(f"📥 Downloading Kaggle dataset: {dataset_ref}")
    os.makedirs(download_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=download_dir, unzip=True)

    return f"Dataset {dataset_ref} downloaded to {download_dir}"

def kaggle_search_tool(query: str) -> list:
    """
    Tool: kaggle_search_tool
    Purpose: Search Kaggle datasets by keyword.
    Inputs:
        query (str): Search term (e.g., 'housing prices', 'climate data', etc.)
    Output: List of dataset references and titles
    """
    print(f"🔎 Searching Kaggle datasets for: {query}")
    api = KaggleApi()
    api.authenticate()
    results = api.dataset_list(search=query)
    return [f"{d.ref} — {d.title}" for d in results[:10]]


# 🚀 Main Dedalus Runner --------------------------------------------------------

def main():
    client = Dedalus()
    runner = DedalusRunner(client)

    result = runner.run(
        input="""
Use the `kaggle_download_tool` to fetch a dataset, then use `autogluon_tool` to train a model on it.
Use the `kaggle_search_tool` to search for a dataset.       

Goal: Determine which wine variety yields higher average score.
""",
        model=["openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514"],
        tools=[kaggle_download_tool, autogluon_tool, kaggle_search_tool],
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
