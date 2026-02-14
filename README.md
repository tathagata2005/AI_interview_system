# FILE: README.md
# AI Interview System

An AI-assisted interview platform built with Flask, SQLite, Gemini question generation, browser microphone speech-to-text, and a TF-IDF + classifier evaluation pipeline.

## Features
- Domain-based interview flow (`HR`, `Technical`, `Behavioral`)
- AI-generated interview questions
- Text answer submission with browser voice input auto-fill
- ML-based answer scoring (`0-100`) from trained artifacts
- Result page with score, feedback, strengths, and weaknesses

## Tech Stack
- Backend: `Flask`
- Database: `SQLite` (via `Flask-SQLAlchemy`)
- ML: `TF-IDF` + `LogisticRegression` / `LinearSVC` (best model auto-selected at train time)
- AI API: `Gemini` (question generation)
- Speech-to-Text: Browser `Web Speech API` for microphone input

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
|-- artifacts/
```

## Setup
1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` from `.env.example` and set your key values.

## Environment Variables
Use `.env` in project root:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
SECRET_KEY=change-this
DATABASE_URL=sqlite:///interview.db
TFIDF_PATH=artifacts/tfidf.pkl
MODEL_PATH=artifacts/model.pkl
MAX_CONTENT_LENGTH=16777216
```

## Dataset Quality Check
```bash
python -m ml.check_data_quality --data "qnagrading_dataset.csv" --text-col answer --label-col label --domain-col domain
```

## Train Model
The training script compares:
- input modes: `answer_only` and `domain_question_answer`
- estimators: `LogisticRegression` and `LinearSVC`

It saves the best model automatically.

```bash
python -m ml.train_model --data "qnagrading_dataset.csv" --out-dir artifacts
```

Generated artifacts:
- `artifacts/tfidf.pkl`
- `artifacts/model.pkl`
- `artifacts/metrics.json`
- `artifacts/model_meta.json`

## Run Application
```bash
python app.py
```
Open: `http://127.0.0.1:5000`

## Demo Flow
1. Select domain
2. Start interview (question generated)
3. Speak into mic to auto-fill answer (or type manually), then submit
4. View score and feedback

## Current Model Snapshot
- Multi-class labels: `0`, `1`, `2`
- Best observed baseline: Accuracy around `0.76 - 0.78`, Macro-F1 around `0.75 - 0.76` (synthetic dataset)

## Limitations
- Model trained on synthetic/small dataset
- Feedback text is rule-based after score conversion
- Browser speech recognition quality depends on browser and microphone quality

## Future Improvements
- Train on larger real labeled interview-answer datasets
- Add toxicity and relevance penalty layer
- Add domain-specific calibration
- Add charts/dashboard analytics
