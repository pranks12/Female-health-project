import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agents.kb_loader import load_knowledge_base, get_kb_summary
from agents.agent3 import chat, get_status

app = FastAPI(title="Fern Health")

# Load KB on startup
load_knowledge_base()


class ChatRequest(BaseModel):
    message: str
    history: list = []


class ChatResponse(BaseModel):
    response: str
    sources: list


@app.get("/api/health")
def health_check():
    kb = get_kb_summary()
    return {"status": "ok", "kb_loaded": kb["loaded"], "kb_files": kb["files"], "kb_topics": kb["topics"]}


@app.get("/api/agent3/status")
def agent_status():
    return get_status()


@app.post("/api/agent3/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    result = chat(req.message, req.history)
    return ChatResponse(response=result["response"], sources=result["sources"])


# Serve frontend — must be last
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
