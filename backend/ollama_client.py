from typing import Any

import httpx


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:3b"

class OllamaConnectionError(RuntimeError):
    """Raised when the backend cannot communicate with Ollama."""


async def send_ollama_prompt(
    prompt: str,
    temperature: float,
) -> str:
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "keep_alive": "10m",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {
            "temperature": temperature,
            "num_predict": 120,
            "num_ctx": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                OLLAMA_CHAT_URL,
                json=payload,
            )
            response.raise_for_status()

    except httpx.HTTPError as error:
        raise OllamaConnectionError(
            "Could not connect to Ollama. "
            "Make sure Ollama is installed and running."
        ) from error

    data = response.json()
    content = data.get("message", {}).get("content")

    if not isinstance(content, str) or not content.strip():
        raise OllamaConnectionError(
            "Ollama returned an empty or invalid response."
        )

    return content.strip()


async def generate_interviewer_response(
    candidate_message: str,
) -> str:
    prompt = f"""
You are a concise software engineering interviewer.

The candidate is solving the Two Sum problem.

Candidate explanation:
{candidate_message}

Ask exactly one short follow-up question.

Rules:
- Do not reveal the solution.
- Do not write code.
- Ask about reasoning, complexity, or edge cases.
- Return only the question.
- Use no more than 30 words.
""".strip()

    result = await send_ollama_prompt(
        prompt=prompt,
        temperature=0.3,
    )

    return result


async def generate_code_review(
    code: str,
    candidate_explanation: str,
) -> str:
    prompt = f"""
Review this Two Sum solution briefly.

Candidate explanation:
{candidate_explanation or "No explanation provided."}

Code:
{code}

Return exactly:

Correctness:
Time:
Space:
Strength:
Improvement:

Maximum 80 words.
""".strip()

    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "prompt": prompt,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": 120,
            "num_ctx": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json=payload,
            )
            response.raise_for_status()

    except httpx.HTTPError as error:
        raise OllamaConnectionError(
            "Could not generate the code review."
        ) from error

    data = response.json()
    review = data.get("response")

    if not isinstance(review, str) or not review.strip():
        raise OllamaConnectionError(
            "Ollama returned an empty code review."
        )

    return review.strip()