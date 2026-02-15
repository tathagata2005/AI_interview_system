# FILE: services/question_service.py
from flask import current_app


def _extract_response_text(response) -> str:
    # Prefer direct text field if present.
    text = (getattr(response, "text", "") or "").strip()
    if text:
        return text
    # Fallback: join candidate parts.
    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            value = getattr(part, "text", "") or ""
            if value:
                parts.append(value.strip())
    return "\n".join(parts).strip()


def _parse_question_payload(text: str):
    # Parse strict expected format first.
    if not text:
        return None, None

    normalized = text.strip()
    if "QUESTION:" in normalized and "IDEAL_ANSWER:" in normalized:
        question_part, answer_part = normalized.split("IDEAL_ANSWER:", 1)
        question = question_part.replace("QUESTION:", "").strip()
        ideal_answer = answer_part.strip()
        if question and ideal_answer:
            return question, ideal_answer

    # Fallback parser: first line question, rest ideal answer.
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    return None, None


def _fallback_question_and_answer(domain: str):
    # Local fallback when API is unavailable.
    fallback = {
        "HR": (
            "How do you handle conflict with a coworker in a professional setting?",
            "I address conflict privately, listen first, align on shared goals, and agree on clear next steps. "
            "I then follow up to confirm progress and keep communication respectful and solution-focused.",
        ),
        "Technical": (
            "Explain the difference between a list and a tuple in Python and when to use each.",
            "A list is mutable and suited for dynamic collections where items change. "
            "A tuple is immutable and better for fixed records, safer data sharing, and dictionary keys.",
        ),
        "Behavioral": (
            "Tell me about a time you had to adapt quickly to a major change.",
            "I use STAR: describe the change, my response plan, actions taken, and measurable outcome. "
            "I prioritize communication, fast learning, and iterative adjustment to deliver stable results.",
        ),
    }
    return fallback.get(
        domain,
        (
            "Tell me about yourself and your most relevant experience.",
            "Summarize background, key achievements, and how your skills match the role with specific outcomes.",
        ),
    )


def generate_question_and_ideal_answer(domain: str):
    # Read Gemini settings from app config.
    api_key = current_app.config.get("GEMINI_API_KEY", "").strip()
    model = current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash")

    if api_key:
        try:
            # Lazy import so app can run without Gemini package.
            from google import genai

            client = genai.Client(api_key=api_key)
            # Prompt asks for one question + one ideal answer.
            prompt = f"""
You are an interview coach.
Generate exactly one {domain} interview question and one strong ideal answer.
Return only in this format:
QUESTION: <question>
IDEAL_ANSWER: <answer>
The ideal answer should be concise, professional, and practical.
""".strip()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.7},
            )
            # Parse model output into (question, ideal_answer).
            text = _extract_response_text(response)
            question, ideal_answer = _parse_question_payload(text)
            if question and ideal_answer:
                return question, ideal_answer
            # Log parse issue and fallback.
            current_app.logger.warning("Gemini response parse failed. Raw text: %s", text[:500])
        except Exception as exc:
            # Log API/runtime error and fallback.
            current_app.logger.exception("Gemini question generation failed: %s", exc)
    else:
        # No API key configured.
        current_app.logger.warning("GEMINI_API_KEY is empty. Using fallback question.")

    # Always return a valid pair.
    return _fallback_question_and_answer(domain)


def generate_question(domain: str) -> str:
    # Helper used when only question text is needed.
    question, _ = generate_question_and_ideal_answer(domain)
    return question
