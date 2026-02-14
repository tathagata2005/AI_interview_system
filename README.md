# FILE: README.md
# AI Interview System

AI-assisted interview practice platform built with Flask, SQLite, Gemini question generation, browser mic input, and TF-IDF + classifier-based scoring.

## Current Features
- Domain selection: `HR`, `Technical`, `Behavioral`
- Configurable interview length: `1`, `5`, `10`, `15` questions
- Configurable timer per question: `90s` or `120s`
- AI-generated question + ideal answer per question (Gemini)
- Browser microphone dictation (Web Speech API) to fill answer box
- Auto-submit on timer timeout
- Per-question scoring, strengths, weaknesses, and feedback
- Final dashboard with:
  - total score
  - average score
  - performance badge
  - per-question breakdown cards
- Interview history page (`/history`) with delete support

## Tech Stack
- Backend: `Flask`
- Database: `SQLite` via `Flask-SQLAlchemy`
- ML: `TF-IDF` + `LogisticRegression` / `LinearSVC`
- AI API: `Gemini` (`google-genai`)
- Voice Input: Browser `Web Speech API` (no server-side audio upload)

## Project Structure
```text
.
|-- app.py
|-- config.py
|-- models.py
|-- requirements.txt
|-- routes/
|-- services/
|-- ml/
|-- templates/
|-- static/
|-- artifacts/
```

## Setup
1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` in project root.

## Environment Variables (`.env`)
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
SECRET_KEY=change-this
DATABASE_URL=sqlite:///interview.db
TFIDF_PATH=artifacts/tfidf.pkl
MODEL_PATH=artifacts/model.pkl
MAX_CONTENT_LENGTH=16777216
```

## Train / Retrain Model
Dataset quality check:
```bash
python -m ml.check_data_quality --data "qnagrading_dataset.csv" --text-col answer --label-col label --domain-col domain
```

Train model:
```bash
python -m ml.train_model --data "qnagrading_dataset.csv" --out-dir artifacts
```

Generated artifacts:
- `artifacts/tfidf.pkl`
- `artifacts/model.pkl`
- `artifacts/metrics.json`
- `artifacts/model_meta.json`

## Run App
```bash
python app.py
```
Open: `http://127.0.0.1:5000`

## User Flow
1. Open home page.
2. Select domain, question count, and timer.
3. Answer each question (type or mic dictation).
4. Submit each answer or let timer auto-submit.
5. View final evaluation dashboard.
6. View/delete past attempts on `/history`.

## Scoring (Current)
- Uses trained TF-IDF + classifier output as primary scoring logic.
- Produces score in range `0–100`.
- Maps score bands to feedback/strengths/weaknesses.

## Known Constraints
- Quality depends on training dataset quality and label design.
- Gemini usage depends on API quota/rate limits.
- Browser mic dictation quality varies by browser, mic, and noise.

## Next Recommended Improvement
- Retrain with question-aware labeled data:
  - `question + ideal_answer + user_answer -> label`
- This improves correctness for "same answer on different question" cases.
