from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

from agents.agent3 import router as agent3_router
from agents.kb_loader import load_kb

app = FastAPI(title="Fern Health")

app.include_router(agent3_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health_check():
    kb_loaded = bool(load_kb())
    return {"status": "ok", "agents": ["agent3"], "kb_loaded": kb_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
