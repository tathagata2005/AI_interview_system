# FILE: ml/train_model.py
import argparse
import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.text_utils import clean_text

DEFAULT_MODES = ("answer_only", "domain_question_answer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF text classifier and auto-pick best mode/model."
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--domain-col", default="domain", help="Domain column name.")
    parser.add_argument("--question-col", default="question", help="Question column name.")
    parser.add_argument("--text-col", default="answer", help="Answer text column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory for model files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state.")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for grid search.")
    parser.add_argument(
        "--modes",
        default="answer_only,domain_question_answer",
        help="Comma-separated modes: answer_only,domain_question_answer",
    )
    return parser.parse_args()


def _normalize_label(label):
    if isinstance(label, str):
        stripped = label.strip()
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except ValueError:
            return stripped
    return label


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def load_dataframe(data_path: str, domain_col: str, question_col: str, text_col: str, label_col: str):
    df = pd.read_csv(data_path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    use_cols = [text_col, label_col]
    if domain_col in df.columns:
        use_cols.append(domain_col)
    if question_col in df.columns:
        use_cols.append(question_col)
    df = df[use_cols].dropna(subset=[text_col, label_col]).copy()

    df["clean_answer"] = df[text_col].astype(str).map(clean_text)
    df["clean_domain"] = df[domain_col].astype(str).map(clean_text) if domain_col in df.columns else ""
    df["clean_question"] = df[question_col].astype(str).map(clean_text) if question_col in df.columns else ""
    df["label"] = df[label_col].map(_normalize_label)
    df = df[df["clean_answer"] != ""]
    return df


def build_texts(df: pd.DataFrame, mode: str):
    if mode == "answer_only":
        texts = df["clean_answer"].astype(str)
    elif mode == "domain_question_answer":
        has_domain = "clean_domain" in df.columns
        has_question = "clean_question" in df.columns
        if has_domain and has_question:
            texts = (
                "domain "
                + df["clean_domain"].astype(str)
                + " question "
                + df["clean_question"].astype(str)
                + " answer "
                + df["clean_answer"].astype(str)
            )
        elif has_question:
            texts = "question " + df["clean_question"].astype(str) + " answer " + df["clean_answer"].astype(str)
        else:
            texts = df["clean_answer"].astype(str)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return texts.str.replace(r"\s+", " ", regex=True).str.strip()


def _make_grid(random_state: int):
    tfidf_params = {
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__max_features": [10000, 20000],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.95],
        "tfidf__sublinear_tf": [True, False],
    }
    return [
        {
            **tfidf_params,
            "clf": [LogisticRegression(max_iter=4000, random_state=random_state)],
            "clf__C": [0.5, 1.0, 2.0, 4.0],
            "clf__class_weight": [None, "balanced"],
        },
        {
            **tfidf_params,
            "clf": [LinearSVC(random_state=random_state)],
            "clf__C": [0.5, 1.0, 2.0, 4.0],
            "clf__class_weight": [None, "balanced"],
        },
    ]


def train_for_mode(x_train, y_train, cv_folds: int, random_state: int):
    pipeline = Pipeline(steps=[("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=_make_grid(random_state=random_state),
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(x_train, y_train)
    return grid


def evaluate_model(best_pipeline, x_test, y_test):
    y_pred = best_pipeline.predict(x_test)
    ordered_labels = sorted(pd.Series(y_test).unique().tolist())
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "labels": [
            int(x) if isinstance(x, (int, float)) and float(x).is_integer() else str(x)
            for x in ordered_labels
        ],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=ordered_labels).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    return metrics


def save_artifacts(tfidf, model, metrics, model_meta, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tfidf_path = os.path.join(out_dir, "tfidf.pkl")
    model_path = os.path.join(out_dir, "model.pkl")
    metrics_path = os.path.join(out_dir, "metrics.json")
    meta_path = os.path.join(out_dir, "model_meta.json")

    with open(tfidf_path, "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(model_meta, meta_file, indent=2)

    return tfidf_path, model_path, metrics_path, meta_path


def main():
    args = parse_args()
    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    modes = [m for m in requested_modes if m in DEFAULT_MODES]
    if not modes:
        raise ValueError("No valid modes selected.")

    df = load_dataframe(
        data_path=args.data,
        domain_col=args.domain_col,
        question_col=args.question_col,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    stratify_labels = df["label"] if df["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_labels,
    )

    all_results = []
    best_result = None
    for mode in modes:
        x_train = build_texts(train_df, mode=mode)
        x_test = build_texts(test_df, mode=mode)
        y_train = train_df["label"]
        y_test = test_df["label"]

        grid = train_for_mode(
            x_train=x_train,
            y_train=y_train,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        best_pipeline = grid.best_estimator_
        mode_metrics = evaluate_model(best_pipeline=best_pipeline, x_test=x_test, y_test=y_test)
        mode_result = {
            "mode": mode,
            "best_cv_score_f1_macro": float(grid.best_score_),
            "best_params": _json_safe(grid.best_params_),
            "estimator": best_pipeline.named_steps["clf"].__class__.__name__,
            **mode_metrics,
        }
        all_results.append(mode_result)

        if best_result is None:
            best_result = {"pipeline": best_pipeline, **mode_result}
        else:
            current_key = (mode_result["f1_macro"], mode_result["accuracy"])
            best_key = (best_result["f1_macro"], best_result["accuracy"])
            if current_key > best_key:
                best_result = {"pipeline": best_pipeline, **mode_result}

    metrics = {
        "selected_mode": best_result["mode"],
        "selected_estimator": best_result["estimator"],
        "accuracy": best_result["accuracy"],
        "f1_macro": best_result["f1_macro"],
        "labels": best_result["labels"],
        "confusion_matrix": best_result["confusion_matrix"],
        "classification_report": best_result["classification_report"],
        "best_cv_score_f1_macro": best_result["best_cv_score_f1_macro"],
        "best_params": best_result["best_params"],
        "all_mode_results": all_results,
    }
    model_meta = {
        "input_mode": best_result["mode"],
        "estimator": best_result["estimator"],
        "labels": best_result["labels"],
    }

    tfidf = best_result["pipeline"].named_steps["tfidf"]
    model = best_result["pipeline"].named_steps["clf"]
    tfidf_path, model_path, metrics_path, meta_path = save_artifacts(
        tfidf=tfidf,
        model=model,
        metrics=metrics,
        model_meta=model_meta,
        out_dir=args.out_dir,
    )

    print(f"Saved TF-IDF: {tfidf_path}")
    print(f"Saved Model: {model_path}")
    print(f"Saved Metrics: {metrics_path}")
    print(f"Saved Model Meta: {meta_path}")
    print(f"Selected Mode: {metrics['selected_mode']}")
    print(f"Selected Estimator: {metrics['selected_estimator']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Best CV Macro F1: {metrics['best_cv_score_f1_macro']:.4f}")
    print(f"Best Params: {metrics['best_params']}")
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
