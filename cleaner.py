# FILE: find_inconsistent_ideals.py
import re
import pandas as pd

DATA_PATH = "similarity_dataset.csv"


def normalize(text: str) -> str:
    text = "" if pd.isna(text) else str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    required = {"question", "ideal_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["_q"] = df["question"].map(normalize)
    df["_ideal"] = df["ideal_answer"].map(normalize)

    grouped = df.groupby("_q")["_ideal"].nunique()
    bad_questions = grouped[grouped > 1].index.tolist()

    print(f"Inconsistent questions found: {len(bad_questions)}\n")

    for i, q in enumerate(bad_questions, start=1):
        ideals = (
            df.loc[df["_q"] == q, "ideal_answer"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )
        print(f"{i}. Question:")
        original_q = df.loc[df["_q"] == q, "question"].iloc[0]
        print(f"   {original_q}")
        print("   Ideal answer variants:")
        for j, ideal in enumerate(ideals, start=1):
            print(f"   [{j}] {ideal}")
        print("-" * 100)


if __name__ == "__main__":
    main()
