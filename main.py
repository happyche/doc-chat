# -*- coding: utf-8 -*-
"""FastAPI application – RAG chatbot with document management and model config."""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config
from rag_engine import RAGEngine

app = FastAPI(title="RAG 智能文档助手")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

engine = RAGEngine()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    history: Optional[list[dict]] = None


class SettingsRequest(BaseModel):
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def generate():
        try:
            for chunk in engine.query(req.question, req.history):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename.endswith(".md"):
            results.append({"filename": file.filename, "error": "仅支持 .md 格式文件"})
            continue

        try:
            raw = await file.read()
            text = raw.decode("utf-8")
            (config.UPLOADS_DIR / file.filename).write_text(text, encoding="utf-8")
            num_chunks = engine.add_document(text, file.filename)
            results.append({"filename": file.filename, "chunks": num_chunks, "status": "success"})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}


@app.get("/api/documents")
async def list_documents():
    return {"documents": engine.list_documents()}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    engine.remove_document(filename)
    fpath = config.UPLOADS_DIR / filename
    if fpath.exists():
        fpath.unlink()
    return {"status": "success"}


@app.get("/api/settings")
async def get_settings():
    settings = config.load_settings()
    safe = dict(settings)
    key = safe.get("llm_api_key", "")
    if len(key) > 12:
        safe["llm_api_key_display"] = key[:8] + "***" + key[-4:]
    else:
        safe["llm_api_key_display"] = "***" if key else ""
    return safe


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    current = config.load_settings()
    current.update(req.model_dump(exclude_none=True))
    config.save_settings(current)
    engine.reload_settings()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
