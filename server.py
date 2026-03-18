import uvicorn
from classification_server import app

if __name__ == "__main__":
    uvicorn.run(
        "classification_server:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        access_log=True,
        log_level="info"
    )