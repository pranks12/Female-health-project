import os
import anthropic
from fastapi import APIRouter
from pydantic import BaseModel
from agents.kb_loader import load_kb

router = APIRouter(prefix="/api/agent3", tags=["agent3"])

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a compassionate and knowledgeable women's health assistant for Fern Health. \
You specialize in treatment options, reproductive health, hormonal health, and general wellness for women. \
Be empathetic, evidence-based, and always recommend consulting a healthcare professional for medical decisions.{kb_context}"""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    kb_context = load_kb()
    system = SYSTEM_PROMPT.format(
        kb_context=f"\n\nKnowledge Base:\n{kb_context}" if kb_context else ""
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": request.message}],
    )

    return ChatResponse(response=message.content[0].text)
