# FILE: services/evaluation_service.py
import json
import os
import pickle
from functools import lru_cache

import numpy as np
from flask import current_app

from ml.text_utils import clean_text


# Load and cache pickled model/vectorizer.
@lru_cache(maxsize=8)
def _load_pickle(path: str):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


# Load optional metadata for quality model input mode.
@lru_cache(maxsize=4)
def _load_quality_meta(model_path: str):
    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return {}


def _token_set(text: str) -> set[str]:
    # Tokenize by whitespace.
    return set(text.split()) if text else set()


def _cosine_from_vectors(a_vec, b_vec) -> float:
    # Cosine similarity for sparse vectors.
    numerator = float(a_vec.multiply(b_vec).sum())
    a_norm = float(np.sqrt(a_vec.multiply(a_vec).sum()))
    b_norm = float(np.sqrt(b_vec.multiply(b_vec).sum()))
    denom = a_norm * b_norm
    if denom == 0.0:
        return 0.0
    return numerator / denom


def _build_similarity_features(ideal_answer: str, answer: str, similarity_tfidf):
    # Build features exactly as used in similarity model training.
    i_text = clean_text(ideal_answer)
    u_text = clean_text(answer)

    i_vec = similarity_tfidf.transform([i_text])
    u_vec = similarity_tfidf.transform([u_text])
    ideal_user_cos = _cosine_from_vectors(i_vec, u_vec)

    i_tokens = _token_set(i_text)
    u_tokens = _token_set(u_text)
    inter = len(i_tokens & u_tokens)
    union = len(i_tokens | u_tokens)

    jaccard = inter / union if union else 0.0
    keyword_overlap = inter / len(i_tokens) if i_tokens else 0.0

    i_len = len(i_text)
    u_len = len(u_text)
    length_ratio = (u_len / i_len) if i_len > 0 else 0.0
    abs_length_delta = abs(u_len - i_len) / max(1, i_len)

    return np.array(
        [[ideal_user_cos, jaccard, keyword_overlap, length_ratio, abs_length_delta]]
    )


def _build_quality_input(answer: str, domain: str, question: str, mode: str) -> str:
    # Prepare text in same format used while training quality model.
    clean_answer = clean_text(answer)
    clean_domain = clean_text(domain)
    clean_question = clean_text(question)

    if mode == "domain_question_answer" and clean_domain and clean_question:
        text = f"domain {clean_domain} question {clean_question} answer {clean_answer}"
    elif mode == "domain_question_answer" and clean_question:
        text = f"question {clean_question} answer {clean_answer}"
    else:
        text = clean_answer

    return " ".join(text.split())


def _to_int_label(value, default: int = 1) -> int:
    # Convert predicted label safely to int.
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _quality_name(label: int) -> str:
    # Map numeric quality class to UI label.
    if label >= 2:
        return "Good"
    if label == 1:
        return "Average"
    return "Poor"


def _feedback_for(correctness: int, quality_label: int) -> str:
    # Select main feedback message.
    if correctness == 0:
        return "Your answer does not sufficiently match the expected response."
    if quality_label >= 2:
        return "Strong and well-structured answer."
    if quality_label == 1:
        return "Correct but could be more detailed."
    return "Correct idea but weak structure."


def _strengths_for(correctness: int, quality_label: int) -> str:
    # Select strengths text.
    if correctness == 0:
        return "Attempted the question."
    if quality_label >= 2:
        return "Clear structure, relevance, and confidence."
    if quality_label == 1:
        return "Core idea is correct."
    return "Captures the basic correct intent."


def _weaknesses_for(correctness: int, quality_label: int) -> str:
    # Select weaknesses text.
    if correctness == 0:
        return "Low alignment with the expected answer."
    if quality_label >= 2:
        return "Could still include measurable outcomes or concise examples."
    if quality_label == 1:
        return "Needs stronger depth and examples."
    return "Needs better clarity and structure."



# MAIN EVALUATION FUNCTION
def evaluate_answer(answer: str, domain: str = "", question: str = "", ideal_answer: str = ""):
    # Read model/vectorizer paths from config.
    similarity_model_path = current_app.config["SIMILARITY_MODEL_PATH"]         #MODEL 1
    similarity_tfidf_path = current_app.config["SIMILARITY_TFIDF_PATH"]
    quality_model_path = current_app.config["QUALITY_MODEL_PATH"]               #MODEL 2
    quality_tfidf_path = current_app.config["QUALITY_TFIDF_PATH"]

    # Default fallback values if inference fails.
    correctness_label = 0
    quality_label = 1

    try:
        # 1) Correctness model: ideal_answer vs user_answer.
        similarity_model = _load_pickle(similarity_model_path)
        similarity_tfidf = _load_pickle(similarity_tfidf_path)
        similarity_features = _build_similarity_features(
            ideal_answer=ideal_answer,
            answer=answer,
            similarity_tfidf=similarity_tfidf,
        )
        raw_label = _to_int_label(similarity_model.predict(similarity_features)[0], default=0)
        correctness_label = 1 if raw_label == 1 else 0
    except Exception:
        # Keep safe default on any model/vectorizer error.
        correctness_label = 0

    try:
        # 2) Quality model: writing quality of the answer text.
        quality_model = _load_pickle(quality_model_path)
        quality_tfidf = _load_pickle(quality_tfidf_path)
        meta = _load_quality_meta(quality_model_path)
        input_mode = meta.get("input_mode", "answer_only")
        quality_input = _build_quality_input(answer=answer, domain=domain, question=question, mode=input_mode)
        if quality_input:
            quality_features = quality_tfidf.transform([quality_input])
            quality_label = _to_int_label(quality_model.predict(quality_features)[0], default=1)
    except Exception:
        # Keep average quality if quality inference fails.
        quality_label = 1

    # Clamp quality label to expected class range.
    quality_label = max(0, min(2, quality_label))
    quality_score = float(quality_label)

    # Final score rule (0-10 per question).
    if correctness_label == 0:
        final_score = quality_score * 2.0
    else:
        final_score = 6.0 + (quality_score * 2.0)
    final_score = round(max(0.0, min(10.0, final_score)), 2)

    # Build display labels and feedback text.
    quality_text = _quality_name(quality_label)
    correctness_text = "Correct" if correctness_label == 1 else "Incorrect"
    feedback = _feedback_for(correctness_label, quality_label)
    strengths = _strengths_for(correctness_label, quality_label)
    weaknesses = _weaknesses_for(correctness_label, quality_label)

    return {
        "score": final_score,
        "correctness_label": correctness_label,
        "correctness_text": correctness_text,
        "quality_label": quality_label,
        "quality_text": quality_text,
        "feedback": feedback,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
