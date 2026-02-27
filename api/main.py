from fastapi import FastAPI

from api.routers import pets

app = FastAPI(
    title="寵物鼻紋辨認 API",
    description="透過鼻紋辨認寵物身份，支援狗（Phase 1）和貓（Phase 2）",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


app.include_router(pets.router, prefix="/api/v1")
