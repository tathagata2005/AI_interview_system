# FILE: app.py
from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import inspect, text

from config import Config
from models import db
from routes.main_routes import main_bp

load_dotenv()


def ensure_schema_updates():
    inspector = inspect(db.engine)
    if "interview_results" not in inspector.get_table_names():
        return
    existing_columns = {col["name"] for col in inspector.get_columns("interview_results")}
    if "ideal_answer" not in existing_columns:
        db.session.execute(text("ALTER TABLE interview_results ADD COLUMN ideal_answer TEXT"))
        db.session.commit()
    if "interview_session_id" not in existing_columns:
        db.session.execute(text("ALTER TABLE interview_results ADD COLUMN interview_session_id TEXT"))
        db.session.commit()
    if "timer_seconds" not in existing_columns:
        db.session.execute(text("ALTER TABLE interview_results ADD COLUMN timer_seconds INTEGER"))
        db.session.commit()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        ensure_schema_updates()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
