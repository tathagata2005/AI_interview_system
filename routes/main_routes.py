# FILE: routes/main_routes.py
from collections import OrderedDict
from uuid import uuid4

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import desc

from models import InterviewResult, db
from services.evaluation_service import evaluate_answer
from services.question_service import generate_question_and_ideal_answer


main_bp = Blueprint("main", __name__)


def _overall_feedback(avg_score: float) -> str:
    if avg_score >= 80:
        return "Excellent overall performance. Your answers were clear, relevant, and well-structured."
    if avg_score >= 65:
        return "Good overall performance. Add more concrete examples and measurable outcomes."
    if avg_score >= 50:
        return "Average overall performance. Improve depth, structure, and role-specific clarity."
    return "Needs improvement overall. Focus on structure, relevance, and specific examples in each answer."


def _performance_label(avg_score: float):
    if avg_score >= 80:
        return "Excellent", "excellent"
    if avg_score >= 65:
        return "Good", "good"
    if avg_score >= 50:
        return "Average", "average"
    return "Needs Improvement", "needs-improvement"


@main_bp.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@main_bp.route("/history", methods=["GET"])
def history():
    rows = InterviewResult.query.order_by(desc(InterviewResult.created_at)).all()
    grouped = OrderedDict()
    for row in rows:
        group_key = row.interview_session_id or f"legacy-{row.id}"
        if group_key not in grouped:
            grouped[group_key] = {
                "session_id": group_key,
                "domain": row.domain,
                "timer_seconds": row.timer_seconds,
                "attempted_at": row.created_at,
                "question_count": 0,
                "total_score": 0.0,
            }
        grouped[group_key]["question_count"] += 1
        grouped[group_key]["total_score"] += float(row.score or 0.0)

    attempts = []
    for item in grouped.values():
        question_count = item["question_count"] or 1
        item["average_score"] = round(item["total_score"] / question_count, 2)
        item["total_score"] = round(item["total_score"], 2)
        attempts.append(item)
    return render_template("history.html", attempts=attempts)


@main_bp.route("/history/delete/<string:session_id>", methods=["POST"])
def delete_history(session_id: str):
    deleted_count = 0
    if session_id.startswith("legacy-"):
        legacy_id = session_id.replace("legacy-", "", 1)
        if legacy_id.isdigit():
            row = InterviewResult.query.get(int(legacy_id))
            if row:
                db.session.delete(row)
                deleted_count = 1
    else:
        rows = InterviewResult.query.filter_by(interview_session_id=session_id).all()
        for row in rows:
            db.session.delete(row)
            deleted_count += 1

    if deleted_count > 0:
        db.session.commit()
        flash("History deleted successfully.", "success")
    else:
        flash("No matching history found.", "error")
    return redirect(url_for("main.history"))


@main_bp.route("/start", methods=["POST"])
def start_interview():
    domain = request.form.get("domain", "").strip()
    question_count_raw = request.form.get("question_count", "5").strip()
    timer_raw = request.form.get("timer_seconds", "90").strip()
    if domain not in {"HR", "Technical", "Behavioral"}:
        flash("Please select a valid interview domain.", "error")
        return redirect(url_for("main.home"))
    if question_count_raw not in {"1", "5", "10", "15"}:
        flash("Please select a valid number of questions.", "error")
        return redirect(url_for("main.home"))
    if timer_raw not in {"90", "120"}:
        flash("Please select a valid timer option.", "error")
        return redirect(url_for("main.home"))

    total_questions = int(question_count_raw)
    timer_seconds = int(timer_raw)
    question, ideal_answer = generate_question_and_ideal_answer(domain)
    session["interview_session_id"] = uuid4().hex
    session["domain"] = domain
    session["total_questions"] = total_questions
    session["timer_seconds"] = timer_seconds
    session["current_index"] = 1
    session["answers"] = []
    session["question"] = question
    session["ideal_answer"] = ideal_answer
    return render_template(
        "interview.html",
        domain=domain,
        question=question,
        current_index=1,
        total_questions=total_questions,
        timer_seconds=timer_seconds,
    )


@main_bp.route("/submit", methods=["POST"])
def submit_answer():
    domain = session.get("domain")
    question = session.get("question")
    ideal_answer = session.get("ideal_answer", "")
    interview_session_id = session.get("interview_session_id", "")
    total_questions = int(session.get("total_questions", 0) or 0)
    timer_seconds = int(session.get("timer_seconds", 90) or 90)
    current_index = int(session.get("current_index", 0) or 0)
    answer = request.form.get("answer", "").strip()
    auto_timeout = request.form.get("auto_timeout", "").strip() == "1"

    if not domain or not question or not total_questions or not current_index:
        flash("Session expired. Please start again.", "error")
        return redirect(url_for("main.home"))

    if not answer:
        if auto_timeout:
            answer = "No answer provided (time limit exceeded)."
        else:
            flash("Please provide your answer text.", "error")
            return render_template(
                "interview.html",
                domain=domain,
                question=question,
                current_index=current_index,
                total_questions=total_questions,
                timer_seconds=timer_seconds,
            )

    score, feedback, strengths, weaknesses = evaluate_answer(
        answer=answer,
        domain=domain,
        question=question,
        ideal_answer=ideal_answer,
    )

    result = InterviewResult(
        interview_session_id=interview_session_id,
        domain=domain,
        timer_seconds=timer_seconds,
        question=question,
        ideal_answer=ideal_answer,
        answer_text=answer,
        score=score,
        feedback=feedback,
        strengths=strengths,
        weaknesses=weaknesses,
    )
    db.session.add(result)
    db.session.commit()

    answers = session.get("answers", [])
    answers.append(
        {
            "question_no": current_index,
            "question": question,
            "ideal_answer": ideal_answer,
            "answer_text": answer,
            "score": score,
            "feedback": feedback,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
    )
    session["answers"] = answers

    if current_index < total_questions:
        next_index = current_index + 1
        next_question, next_ideal_answer = generate_question_and_ideal_answer(domain)
        session["current_index"] = next_index
        session["question"] = next_question
        session["ideal_answer"] = next_ideal_answer
        return render_template(
            "interview.html",
            domain=domain,
            question=next_question,
            current_index=next_index,
            total_questions=total_questions,
            timer_seconds=timer_seconds,
        )

    total_score = round(sum(item["score"] for item in answers), 2)
    avg_score = round(total_score / len(answers), 2) if answers else 0.0
    overall_feedback = _overall_feedback(avg_score)
    performance_label, performance_tone = _performance_label(avg_score)

    summary = {
        "domain": domain,
        "total_questions": total_questions,
        "timer_seconds": timer_seconds,
        "total_score": total_score,
        "average_score": avg_score,
        "overall_feedback": overall_feedback,
        "performance_label": performance_label,
        "performance_tone": performance_tone,
        "rows": answers,
    }

    session.pop("domain", None)
    session.pop("interview_session_id", None)
    session.pop("question", None)
    session.pop("ideal_answer", None)
    session.pop("total_questions", None)
    session.pop("timer_seconds", None)
    session.pop("current_index", None)
    session.pop("answers", None)

    return render_template("result.html", summary=summary)
