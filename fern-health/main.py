from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fern Health")

app.mount("/static", StaticFiles(directory="static"), name="static")

from agents.agent3 import router as agent3_router
app.include_router(agent3_router)


@app.get("/api/health")
def health():
    from agents.kb_loader import KBLoader
    kb = KBLoader("kb/agent3_treatment")
    validation = kb.validate()
    return {
        "status": "ok",
        "agents": ["agent3"],
        "kb_loaded": validation["entries"] > 0,
        "kb_entries": validation["entries"],
        "kb_conditions": validation["conditions"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
