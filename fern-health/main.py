from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Fern Health")

# Agent routers
from agents.agent3 import router as agent3_router

app.include_router(agent3_router)

# Serve static assets (JS, CSS, images etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/health")
def health_check():
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {
        "status": "ok",
        "agents": ["agent3"],
        "kb_loaded": True,
        "live_ai_available": api_key_set,
    }


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
