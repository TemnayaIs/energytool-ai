"""Прогресс локальных прогонов eval: python scripts/progress.py [--once]"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "eval"
STAGES = [
    ("think_best (61ab77f)", "runs_think_best.log"),
    ("think_t06 (exp/think-tuning)", "runs_think_t06.log"),
    ("судья", "runs_judge.log"),
]
LINE = re.compile(r"^\s*(\d+)/(\d+)\s")


def fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def stage_state(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    done = total = 0
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
    finished = "->" in text and "seconds" in text
    started = path.stat().st_ctime
    return done, total, finished, started, text


def render() -> str:
    out = []
    remaining = 0.0
    now = time.time()
    for title, name in STAGES:
        state = stage_state(EVAL / name)
        if state is None:
            out.append(f"  ожидает   {title}")
            continue
        done, total, finished, started, text = state
        if finished or (total and done >= total and name != "runs_judge.log"):
            out.append(f"  готово    {title}")
            continue
        elapsed = now - started
        if total and done:
            eta = elapsed / done * (total - done)
            bar = "#" * (30 * done // total) + "." * (30 - 30 * done // total)
            out.append(f"  идёт      {title}\n            [{bar}] {done}/{total}, прошло {fmt(elapsed)}, осталось ~{fmt(eta)}")
            remaining += eta
        else:
            out.append(f"  идёт      {title}, прошло {fmt(elapsed)} (счётчик ещё не появился)")
    if remaining:
        out.append(f"\n  До конца текущего этапа ~{fmt(remaining)}; следующие этапы идут по очереди после него")
    if (EVAL / "runs_chain.done").exists():
        out.append("\n  Вся цепочка закончена")
    return "\n".join(out)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    once = "--once" in sys.argv
    while True:
        if not once:
            os.system("cls" if os.name == "nt" else "clear")
        print("Прогресс eval (обновление раз в 10 с, Ctrl+C - выход)\n")
        print(render())
        if once:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
