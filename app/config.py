"""Конфигурация веб-прототипа. Секреты и параметры моделей читает energy_rag.Settings из окружения."""

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
    return value or default


KB_PATH = Path(env("KB_PATH", str(BASE_DIR / "data" / "knowledge-base")))
DB_PATH = Path(env("DB_PATH", str(BASE_DIR / "data" / "app.db")))
STATIC_DIR = BASE_DIR / "static"

APP_TITLE = "МТС EnergyTool — ИИ-консультант"
APP_VERSION = "0.2.0"
