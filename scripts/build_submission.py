# -*- coding: utf-8 -*-
"""Собирает submission.zip по требованиям технического регламента.

В корне архива должны лежать solution.py, requirements.txt и при необходимости
дополнительные .py-модули. Вложенные папки платформа не переносит, поэтому
rag_core.py и answers_core.py кладутся рядом с solution.py.

Запуск из корня проекта:
    python scripts/build_submission.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# source -> имя внутри архива. solution.py нужен только httpx, поэтому
# в сабмит кладём минимальный requirements-solution.txt под именем requirements.txt.
FILES = {
    "solution.py": "solution.py",
    "answers_core.py": "answers_core.py",
    "rag_core.py": "rag_core.py",
    "requirements-solution.txt": "requirements.txt",
}
OUTPUT = ROOT / "submission.zip"


def main() -> int:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        print("Не найдены файлы: " + ", ".join(missing), file=sys.stderr)
        return 1

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in FILES.items():
            archive.write(ROOT / source, arcname=arcname)

    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        bad = [name for name in names if "/" in name or "\\" in name]
        if bad:
            print("В архиве есть вложенные пути: " + ", ".join(bad), file=sys.stderr)
            return 1
        print(f"Готово: {OUTPUT}")
        print("Содержимое: " + ", ".join(names))
        print(f"Размер: {OUTPUT.stat().st_size} байт")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
