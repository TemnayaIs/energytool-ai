"""Конфигурация прототипа. Все секреты — только из переменных окружения."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value else value


def number(name: str, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name, str(default)))))
    except (TypeError, ValueError):
        return default


KB_PATH = Path(env("KB_PATH", str(BASE_DIR / "data" / "knowledge-base")))
DB_PATH = Path(env("DB_PATH", str(BASE_DIR / "data" / "app.db")))
STATIC_DIR = BASE_DIR / "static"

LLM_BASE_URL = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_API_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "qwen/qwen3.8-27b")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")

TOP_K = number("TOP_K", 6, 1, 20)
RERANKER_TOP_N = number("RERANKER_TOP_N", 4, 1, 20)
MAX_TOKENS = number("MAX_TOKENS", 900, 128, 4096)
TEMPERATURE = number("TEMPERATURE", 0.1, 0.0, 2.0, float)
MAX_WORKERS = number("MAX_WORKERS", 4, 1, 16)
USE_RERANKER = env("USE_RERANKER", "0") in ("1", "true", "yes", "on")
LLM_TIMEOUT = number("LLM_TIMEOUT", 45, 5, 180)

APP_TITLE = "МТС EnergyTool — ИИ-консультант"
APP_VERSION = "0.1.0"
