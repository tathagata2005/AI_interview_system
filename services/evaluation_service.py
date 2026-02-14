# FILE: services/evaluation_service.py
import json
import os
import pickle
from functools import lru_cache

from flask import current_app
from ml.text_utils import clean_text


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    with open(model_path, "rb") as file_obj:
        return pickle.load(file_obj)


@lru_cache(maxsize=1)
def _load_vectorizer(tfidf_path: str):
    with open(tfidf_path, "rb") as file_obj:
        return pickle.load(file_obj)


@lru_cache(maxsize=1)
def _load_model_meta(model_path: str):
    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return {}


def _build_input_text(answer: str, domain: str = "", question: str = "", mode: str = "answer_only") -> str:
    clean_answer = clean_text(answer)
    clean_domain = clean_text(domain)
    clean_question = clean_text(question)

    if mode == "domain_question_answer" and clean_domain and clean_question:
        combined = f"domain {clean_domain} question {clean_question} answer {clean_answer}"
    elif mode == "domain_question_answer" and clean_question:
        combined = f"question {clean_question} answer {clean_answer}"
    else:
        combined = clean_answer
    return " ".join(combined.split())


def _score_from_distribution(classes, values):
    numeric_classes = []
    try:
        for class_value in classes:
            numeric_classes.append(float(class_value))
    except (TypeError, ValueError):
        return 50.0

    if not numeric_classes:
        return 50.0

    max_class = max(numeric_classes)
    min_class = min(numeric_classes)
    expected_class = sum(cls_value * float(weight) for cls_value, weight in zip(numeric_classes, values))
    if max_class <= min_class:
        return 50.0
    return round(((expected_class - min_class) / (max_class - min_class)) * 100, 2)


def evaluate_answer(answer: str, domain: str = "", question: str = "", ideal_answer: str = ""):
    tfidf_path = current_app.config["TFIDF_PATH"]
    model_path = current_app.config["MODEL_PATH"]
    score = 50.0
    feedback = "Average response. Add more specific examples and measurable impact."
    strengths = "Clear structure."
    weaknesses = "Limited depth and evidence."

    try:
        model_meta = _load_model_meta(model_path)
        input_mode = model_meta.get("input_mode", "answer_only")
        model_input = _build_input_text(answer=answer, domain=domain, question=question, mode=input_mode)
        if not model_input:
            return score, feedback, strengths, weaknesses

        vectorizer = _load_vectorizer(tfidf_path)
        model = _load_model(model_path)
        features = vectorizer.transform([model_input])
        classes = getattr(model, "classes_", [])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            score = _score_from_distribution(classes, probabilities)
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(features)
            values = decision[0] if getattr(decision, "ndim", 1) > 1 else [float(decision[0])]
            if len(values) > 1:
                max_value = max(values)
                shifted = [v - max_value for v in values]
                exp_vals = [pow(2.718281828, v) for v in shifted]
                denom = sum(exp_vals) or 1.0
                pseudo_probs = [v / denom for v in exp_vals]
                score = _score_from_distribution(classes, pseudo_probs)
            else:
                score = 50.0
        elif hasattr(model, "predict"):
            predicted = model.predict(features)[0]
            try:
                numeric_classes = sorted(float(c) for c in classes) if len(classes) > 0 else [0.0, 2.0]
                min_class = numeric_classes[0]
                max_class = numeric_classes[-1]
                if max_class > min_class:
                    score = round(((float(predicted) - min_class) / (max_class - min_class)) * 100, 2)
                else:
                    score = 50.0
            except (TypeError, ValueError):
                score = 50.0

        if score >= 80:
            feedback = "Strong response with clarity and relevant detail."
            strengths = "Relevant examples, concise communication, good confidence."
            weaknesses = "Can still improve by adding quantifiable outcomes."
        elif score >= 60:
            feedback = "Good response, but it can be more specific."
            strengths = "Reasonable structure and relevance."
            weaknesses = "Needs stronger examples and clearer outcomes."
        else:
            feedback = "Response needs improvement in clarity and depth."
            strengths = "Basic attempt to address the question."
            weaknesses = "Lacks structure, examples, and impact."
    except Exception:
        pass

    score = round(max(0.0, min(100.0, score)), 2)
    return score, feedback, strengths, weaknesses
