from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama_client import (OllamaConnectionError, generate_interviewer_response)
from ollama_client import (OllamaConnectionError,generate_code_review,generate_interviewer_response,)



app = FastAPI(
    title="InterviewAI Backend",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=3_000,
    )


class ChatResponse(BaseModel):
    response: str


class CodeReviewRequest(BaseModel):
    code: str = Field(
        min_length=1,
        max_length=20_000,
    )
    explanation: str = Field(
        default="",
        max_length=3_000,
    )


class CodeReviewResponse(BaseModel):
    review: str

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "running",
        "application": "InterviewAI",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = await generate_interviewer_response(
            candidate_message=request.message
        )

        return ChatResponse(response=response)

    except OllamaConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


@app.post("/review", response_model=CodeReviewResponse)
async def review_code(
    request: CodeReviewRequest,
) -> CodeReviewResponse:
    try:
        review = await generate_code_review(
            code=request.code,
            candidate_explanation=request.explanation,
        )

        return CodeReviewResponse(review=review)

    except OllamaConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error