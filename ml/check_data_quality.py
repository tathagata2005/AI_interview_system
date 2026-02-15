# FILE: ml/check_data_quality.py
import argparse
import json
import os
import re

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Check data quality for interview/similarity datasets.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--question-col", default="question", help="Question column name.")
    parser.add_argument("--ideal-col", default="ideal_answer", help="Ideal answer column name.")
    parser.add_argument("--text-col", default="user_answer", help="User answer text column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--domain-col", default="domain", help="Domain column name.")
    parser.add_argument("--out", default="", help="Optional output JSON report path.")
    return parser.parse_args()


def safe_dist(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _normalize_text(value):
    value = "" if pd.isna(value) else str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def main():
    args = parse_args()
    df = pd.read_csv(args.data)

    report = {
        "file": os.path.abspath(args.data),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_counts": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "label_distribution": safe_dist(df, args.label_col),
        "domain_distribution": safe_dist(df, args.domain_col),
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_text_rows": int(df.duplicated(subset=[args.text_col]).sum()) if args.text_col in df.columns else 0,
        "duplicate_question_answer_pairs": int(
            df.duplicated(subset=[args.question_col, args.text_col]).sum()
        ) if args.question_col in df.columns and args.text_col in df.columns else 0,
    }

    checks = []

    if args.question_col not in df.columns:
        checks.append(f"Missing required question column: {args.question_col}")
    if args.ideal_col not in df.columns:
        checks.append(f"Missing required ideal answer column: {args.ideal_col}")
    if args.text_col not in df.columns:
        checks.append(f"Missing required text column: {args.text_col}")
    if args.label_col not in df.columns:
        checks.append(f"Missing required label column: {args.label_col}")

    if args.question_col in df.columns:
        empty_question = int(df[args.question_col].fillna("").astype(str).str.strip().eq("").sum())
        report["empty_question_rows"] = empty_question
        if empty_question > 0:
            checks.append(f"Found {empty_question} empty question rows.")

    if args.ideal_col in df.columns:
        empty_ideal = int(df[args.ideal_col].fillna("").astype(str).str.strip().eq("").sum())
        report["empty_ideal_answer_rows"] = empty_ideal
        if empty_ideal > 0:
            checks.append(f"Found {empty_ideal} empty ideal answer rows.")

    if args.text_col in df.columns:
        empty_text = int(df[args.text_col].fillna("").astype(str).str.strip().eq("").sum())
        report["empty_text_rows"] = empty_text
        if empty_text > 0:
            checks.append(f"Found {empty_text} empty text rows.")

    if args.label_col in df.columns:
        labels = pd.to_numeric(df[args.label_col], errors="coerce")
        unique_labels = sorted(labels.dropna().unique().tolist())
        report["unique_labels"] = [str(x) for x in unique_labels]
        if len(unique_labels) < 2:
            checks.append("Need at least 2 label classes.")
        expected = {0.0, 1.0, 2.0}
        observed = set(unique_labels)
        if observed and not observed.issubset(expected):
            checks.append("Labels should be in {0,1,2} for similarity dataset.")
        label_counts = df[args.label_col].value_counts()
        if len(label_counts) >= 2:
            ratio = float(label_counts.min() / label_counts.max())
            report["label_balance_ratio"] = round(ratio, 4)
            if ratio < 0.5:
                checks.append("Label imbalance detected (minority/majority < 0.5).")
        if len(label_counts) == 3:
            report["balanced_3_class"] = bool(float(label_counts.min() / label_counts.max()) >= 0.8)

    if args.question_col in df.columns and args.ideal_col in df.columns:
        sample = df[[args.question_col, args.ideal_col]].dropna().copy()
        sample["_question"] = sample[args.question_col].map(_normalize_text)
        sample["_ideal"] = sample[args.ideal_col].map(_normalize_text)
        inconsistent = int(sample.groupby("_question")["_ideal"].nunique().gt(1).sum())
        report["question_to_multiple_ideals"] = inconsistent
        if inconsistent > 0:
            checks.append(
                "Some questions map to multiple ideal answers; verify data consistency."
            )

    if args.domain_col in df.columns:
        domain_counts = df[args.domain_col].value_counts()
        if len(domain_counts) > 0:
            ratio = float(domain_counts.min() / domain_counts.max())
            report["domain_balance_ratio"] = round(ratio, 4)
            if ratio < 0.5:
                checks.append("Domain imbalance detected (minority/majority < 0.5).")

    report["warnings"] = checks
    report["ok_for_baseline_training"] = len(checks) == 0

    print(json.dumps(report, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
