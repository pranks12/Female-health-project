from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
import anthropic
import json
import os

load_dotenv()

app = FastAPI(title="Fern Health")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Agent routers will be imported here


@app.get("/api/health")
def health_check():
    return {"status": "ok", "agents": ["agent3"], "kb_loaded": False}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_prompt = (
        "You are Fern, a compassionate and knowledgeable women's health assistant. "
        "You provide supportive, evidence-based guidance on topics like menstrual health, "
        "hormonal balance, fertility, menopause, nutrition, and general wellbeing. "
        "You are warm, non-judgmental, and always recommend consulting a healthcare "
        "provider for medical decisions. Keep responses clear and empathetic."
    )

    def stream_response():
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
