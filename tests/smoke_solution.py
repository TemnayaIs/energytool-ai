"""Smoke-тест solution.py: поднимает mock-прокси и проверяет формат submission.csv.

    python tests/smoke_solution.py data/demo_questions.csv data/knowledge-base.zip
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    questions, kb = map(Path, sys.argv[1:3])
    port = "8911"
    proxy = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.mock_proxy:app", "--port", port, "--log-level", "warning"],
        cwd=ROOT,
    )
    try:
        time.sleep(2)
        with tempfile.TemporaryDirectory() as work:
            shutil.copy(questions, Path(work) / "questions.csv")
            shutil.copy(kb, Path(work) / "knowledge-base.zip")
            env = {**os.environ, "DATA_DIR": work, "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1", "LLM_API_KEY": "test"}
            subprocess.run([sys.executable, str(ROOT / "solution.py")], cwd=work, env=env, check=True)
            with open(Path(work) / "submission.csv", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                rows = list(reader)
            with open(questions, encoding="utf-8-sig", newline="") as handle:
                ids = [row["id"] for row in csv.DictReader(handle)]
            assert header == ["id", "answer"], header
            assert [r[0] for r in rows] == ids, "ID не совпадают"
            assert all(r[1].strip() and len(r[1]) < 10000 for r in rows), "пустой или слишком длинный ответ"
            print(f"OK: {len(rows)} ответов")
    finally:
        proxy.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
