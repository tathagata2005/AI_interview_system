# FILE: README.md
# AI Interview System

AI-assisted interview practice platform built with Flask, SQLite, Gemini question generation, browser mic input, and dual-model ML evaluation.

## screenshots
<img width="1552" height="750" alt="image" src="https://github.com/user-attachments/assets/e61d04a2-db61-4cea-80cf-32499af623f4" />
<img width="1310" height="1077" alt="image" src="https://github.com/user-attachments/assets/c8964358-da96-465d-b560-7cc5ef43b738" />

## Current Features
- Domain selection: `HR`, `Technical`, `Behavioral`
- Configurable interview length: `1`, `5`, `10`, `15` questions
- Configurable timer per question: `90s` or `120s`
- AI-generated question + ideal answer per question (Gemini)
- Browser microphone dictation (Web Speech API) to fill answer box
- Auto-submit on timer timeout
- Dual-model evaluation:
  - Similarity model -> `Correct` / `Incorrect`
  - Quality model -> `Good` / `Average` / `Poor`
- Per-question score, feedback, strengths, and weaknesses
- Final dashboard with:
  - total score
  - average score (`/10`)
  - performance badge
  - per-question correctness badge
  - per-question writing quality badge
  - per-question breakdown cards
- Interview history page (`/history`) with delete support

## Tech Stack
- Backend: `Flask`
- Database: `SQLite` via `Flask-SQLAlchemy`
- ML:
  - Similarity model (binary classifier)
  - Quality model (3-class classifier)
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
MODEL_PATH=artifacts/quality_model.pkl
QUALITY_TFIDF_PATH=artifacts/tfidf.pkl
QUALITY_MODEL_PATH=artifacts/quality_model.pkl
SIMILARITY_TFIDF_PATH=artifacts/similarity_tfidf.pkl
SIMILARITY_MODEL_PATH=artifacts/similarity_model.pkl
MAX_CONTENT_LENGTH=16777216
```

## Train / Retrain Models
### 1) Quality Model 
Dataset quality check (quality dataset):
```bash
python -m ml.check_data_quality --data "qnagrading_dataset.csv" --text-col answer --label-col label --domain-col domain
```

Train quality model:
```bash
python -m ml.train_model --data "qnagrading_dataset.csv" --out-dir artifacts
```

Generated artifacts:
- `artifacts/tfidf.pkl`
- `artifacts/quality_model.pkl`
- `artifacts/metrics.json`
- `artifacts/model_meta.json`

Accuracy of my quality model:
```bash
Accuracy: 0.7667
Macro F1: 0.7598
Best CV Macro F1: 0.7181
Confusion Matrix:
[37, 21, 2]
[9, 41, 10]
[0, 0, 60]
```

### 2) Similarity Model (Binary)
Train similarity model:
```bash
python -m ml.similarity_model_train
```

Generated artifacts:
- `artifacts/similarity_model.pkl`
- `artifacts/similarity_tfidf.pkl`

Accuracy of my similarity model:
```bash
Best Model: svc
Best CV F1 (Binary): 0.712
Accuracy: 0.7681
F1 (Binary): 0.6364
Confusion Matrix:
 [[78 20]
 [12 28]]
Classification Report:
               precision    recall  f1-score   support

           0     0.8667    0.7959    0.8298        98
           1     0.5833    0.7000    0.6364        40

    accuracy                         0.7681       138
   macro avg     0.7250    0.7480    0.7331       138
weighted avg     0.7845    0.7681    0.7737       138
```

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

## Evaluation and Scoring
- Step 1: Similarity model predicts correctness using only:
  - `ideal_answer`
  - `user_answer`
- Step 2: Quality model predicts writing quality:
  - `0 -> Poor`
  - `1 -> Average`
  - `2 -> Good`
- Final per-question score (`0..10`):
  - If `Incorrect`: `quality_score * 2`
  - If `Correct`: `6 + (quality_score * 2)`
  - Score is capped between `0` and `10`

## Known Constraints
- Quality depends on training dataset quality and label design.
- Gemini usage depends on API quota/rate limits.
- Browser mic dictation quality varies by browser, mic, and noise.
