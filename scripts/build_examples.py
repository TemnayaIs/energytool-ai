"""Собирает examples.py — банк примеров «вопрос → эталонный ответ» для подбора в промпт."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    (ROOT.parent / "data" / "demo_questions.csv", ROOT.parent / "data" / "demo_output.csv"),
    (ROOT / "eval" / "extra_questions.csv", ROOT / "eval" / "extra_golden.csv"),
]


def read(path: Path, column: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row[column] for row in csv.DictReader(handle)}


def main() -> int:
    pairs = []
    for qpath, apath in SOURCES:
        if not (qpath.is_file() and apath.is_file()):
            print(f"пропущено: {qpath.name}", file=sys.stderr)
            continue
        questions, answers = read(qpath, "question"), read(apath, "answer")
        pairs += [(questions[i], answers[i]) for i in questions if i in answers]
    lines = ["EXAMPLES = ["]
    for q, a in pairs:
        lines.append(f"    ({q!r},\n     {a!r}),")
    lines.append("]")
    (ROOT / "examples.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"examples.py: {len(pairs)} примеров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
