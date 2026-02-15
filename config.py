# FILE: config.py
import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///interview.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    TFIDF_PATH = os.getenv("TFIDF_PATH", "artifacts/tfidf.pkl")
    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/quality_model.pkl")
    QUALITY_MODEL_PATH = os.getenv("QUALITY_MODEL_PATH", os.getenv("MODEL_PATH", "artifacts/quality_model.pkl"))
    QUALITY_TFIDF_PATH = os.getenv("QUALITY_TFIDF_PATH", os.getenv("TFIDF_PATH", "artifacts/tfidf.pkl"))
    SIMILARITY_MODEL_PATH = os.getenv("SIMILARITY_MODEL_PATH", "artifacts/similarity_model.pkl")
    SIMILARITY_TFIDF_PATH = os.getenv("SIMILARITY_TFIDF_PATH", "artifacts/similarity_tfidf.pkl")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
