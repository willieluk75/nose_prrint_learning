from fastapi import FastAPI
from api.config import settings  # noqa: F401 – re-exported for backwards compat

app = FastAPI(
    title="寵物鼻紋辨認 API",
    description="透過鼻紋辨認寵物身份，支援狗（Phase 1）和貓（Phase 2）",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


from api.routers import pets  # noqa: E402
app.include_router(pets.router, prefix="/api/v1")
