# FILE: models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class InterviewResult(db.Model):
    __tablename__ = "interview_results"

    id = db.Column(db.Integer, primary_key=True)
    interview_session_id = db.Column(db.String(64), nullable=True, index=True)
    domain = db.Column(db.String(50), nullable=False)
    timer_seconds = db.Column(db.Integer, nullable=True)
    question = db.Column(db.Text, nullable=False)
    ideal_answer = db.Column(db.Text, nullable=True)
    answer_text = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    strengths = db.Column(db.Text, nullable=True)
    weaknesses = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
