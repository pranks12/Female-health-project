from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fern Health")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Agent routers will be imported here


@app.get("/api/health")
def health_check():
    return {"status": "ok", "agents": ["agent3"], "kb_loaded": False}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
