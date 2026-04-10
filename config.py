# -*- coding: utf-8 -*-
"""Configuration: loads defaults from .env, runtime overrides from settings.json."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SETTINGS_FILE = ROOT / "settings.json"
UPLOADS_DIR = ROOT / "uploads"
CHROMA_DIR = ROOT / "chroma_db"

DEFAULTS = {
    "llm_api_key": os.getenv("LLM_API_KEY", ""),
    "llm_base_url": os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "llm_model": os.getenv("LLM_MODEL", "qwen-plus"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
    "top_k": int(os.getenv("TOP_K", "5")),
}

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULTS, **saved}
    return dict(DEFAULTS)


def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
