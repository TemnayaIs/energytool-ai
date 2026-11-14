"""Локальная проверка через LM Studio.

Скрипт проверяет, поднят ли OpenAI-совместимый сервер LM Studio, определяет
загруженную модель и запускает прогон eval на ограниченном наборе вопросов.

Использование (сервер LM Studio уже запущен):
    python scripts/local_lmstudio.py --limit 30
    python scripts/local_lmstudio.py --limit 30 --name lm_check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "http://127.0.0.1:1234/v1"


def probe(base_url: str) -> str | None:
    try:
        response = httpx.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as error:  # noqa: BLE001
        print(f"LM Studio недоступен на {base_url}: {type(error).__name__}: {error}")
        return None
    models = [item.get("id", "") for item in data.get("data", []) if item.get("id")]
    if not models:
        print("Сервер отвечает, но список моделей пуст: загрузите модель в LM Studio.")
        return None
    print("Доступные модели:")
    for name in models:
        print(f"  - {name}")
    return models[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--name", default="lm_local")
    parser.add_argument("--questions", default="eval/extra_questions.csv")
    parser.add_argument("--workers", default="2")
    args = parser.parse_args()

    model = probe(args.base_url)
    if not model:
        return 1

    env = {
        **os.environ,
        "LLM_BASE_URL": args.base_url,
        "LLM_API_KEY": os.getenv("LLM_API_KEY", "lm-studio"),
        "LLM_MODEL": model,
        "USE_EMBEDDINGS": "0",
        "USE_RERANKER": "0",
        "MAX_WORKERS": args.workers,
        "ANSWER_TEMPERATURE": "0.0",
    }
    command = [
        sys.executable,
        "eval/run_eval.py",
        "gen",
        "--name",
        args.name,
        "--questions",
        args.questions,
        "--limit",
        str(args.limit),
        "--workers",
        args.workers,
    ]
    print(f"\nЗапуск: {' '.join(command)}")
    print(f"Модель: {model}, вопросов: {args.limit}\n")
    result = subprocess.run(command, cwd=ROOT, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
PY