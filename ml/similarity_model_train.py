# FILE: ml/similarity_model_train.py
import pickle
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def _rowwise_cosine(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    numerator = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    a_norm = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    b_norm = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    denom = a_norm * b_norm
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0)


def extract_features(df: pd.DataFrame, vectorizer: TfidfVectorizer | None = None, fit: bool = True):
    ideal = df["ideal_answer_clean"].astype(str).tolist()
    user = df["user_answer_clean"].astype(str).tolist()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    if fit:
        vectorizer.fit(ideal + user)

    ideal_tfidf = vectorizer.transform(ideal)
    user_tfidf = vectorizer.transform(user)

    ideal_user_cos = _rowwise_cosine(ideal_tfidf, user_tfidf)

    jaccard_scores = []
    keyword_overlap_scores = []
    length_ratios = []
    abs_length_delta = []

    for i_text, u_text in zip(ideal, user):
        i_tokens = _token_set(i_text)
        u_tokens = _token_set(u_text)

        inter = len(i_tokens & u_tokens)
        union = len(i_tokens | u_tokens)

        jaccard = inter / union if union else 0.0
        overlap = inter / len(i_tokens) if i_tokens else 0.0

        i_len = len(i_text)
        u_len = len(u_text)
        length_ratio = u_len / i_len if i_len > 0 else 0.0
        length_delta = abs(u_len - i_len) / max(1, i_len)

        jaccard_scores.append(jaccard)
        keyword_overlap_scores.append(overlap)
        length_ratios.append(length_ratio)
        abs_length_delta.append(length_delta)

    X = np.column_stack(
        [
            ideal_user_cos,
            np.array(jaccard_scores),
            np.array(keyword_overlap_scores),
            np.array(length_ratios),
            np.array(abs_length_delta),
        ]
    )
    return X, vectorizer


def train_model(data_path: str = "similarity_dataset.csv", out_dir: str = "artifacts"):
    df = pd.read_csv(data_path)

    required_cols = {"question", "ideal_answer", "user_answer", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["question", "ideal_answer", "user_answer", "label"]).copy()
    df["ideal_answer_clean"] = df["ideal_answer"].apply(clean_text)
    df["user_answer_clean"] = df["user_answer"].apply(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Binary mapping:
    # 2 -> 1 (Correct)
    # 1 -> 0 (Incorrect)
    # 0 -> 0 (Incorrect)
    df["label_binary"] = np.where(df["label"] == 2, 1, 0)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_binary"],
    )

    X_train, vectorizer = extract_features(train_df, vectorizer=None, fit=True)
    X_test, _ = extract_features(test_df, vectorizer=vectorizer, fit=False)
    y_train = train_df["label_binary"].values
    y_test = test_df["label_binary"].values

    candidates = {
        "logreg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
                ]
            ),
            {"clf__C": [0.5, 1.0, 2.0, 4.0]},
        ),
        "svc": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(class_weight="balanced")),
                ]
            ),
            {"clf__C": [0.5, 1.0, 2.0, 4.0], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale"]},
        ),
        "rf": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 12], "min_samples_leaf": [1, 2, 4]},
        ),
    }

    best_name = None
    best_search = None
    best_f1 = -1.0

    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        if search.best_score_ > best_f1:
            best_f1 = float(search.best_score_)
            best_search = search
            best_name = name

    best_model = best_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best Model:", best_name)
    print("Best CV F1 (Binary):", round(best_f1, 4))
    print("Best Params:", best_search.best_params_)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 (Binary):", round(f1_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "similarity_model.pkl"
    tfidf_path = out_path / "similarity_tfidf.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(tfidf_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model: {model_path}")
    print(f"Saved tfidf: {tfidf_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "similarity_dataset.csv"
    artifacts_dir = project_root / "artifacts"
    train_model(str(dataset_path), str(artifacts_dir))
