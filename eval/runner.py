"""Раннер очереди локальных прогонов: выполняет задания из eval/queue/.

Запуск (один раз, держите окно открытым):
    python eval/runner.py

Задание — файл eval/queue/<имя>.job с ОДНОЙ строкой аргументов для run_eval.py, например:
    gen --name v03 --vs v02
Вывод пишется в eval/queue/<имя>.log, после выполнения файл переименовывается в .done.
Ради безопасности выполняются только подкоманды run_eval.py: gen, zip, judge.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "eval" / "queue"
ALLOWED = {"gen", "zip", "judge"}


def run_job(job: Path) -> None:
    log = job.with_suffix(".log")
    try:
        args = shlex.split(job.read_text(encoding="utf-8-sig").strip(), posix=True)
    except ValueError as error:
        log.write_text(f"Ошибка разбора задания: {error}\n", encoding="utf-8")
        job.rename(job.with_suffix(".failed"))
        return
    if not args or args[0] not in ALLOWED:
        log.write_text(f"Отклонено: разрешены только {sorted(ALLOWED)}\n", encoding="utf-8")
        job.rename(job.with_suffix(".failed"))
        return
    print(f"[{time.strftime('%H:%M:%S')}] {job.name}: run_eval.py {' '.join(args)}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(f"=== start {time.strftime('%H:%M:%S')}: {' '.join(args)}\n")
        handle.flush()
        code = subprocess.call(
            [sys.executable, "-u", str(ROOT / "eval" / "run_eval.py"), *args],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        handle.write(f"=== exit {code} за {time.monotonic() - started:.0f} с\n")
    job.rename(job.with_suffix(".done" if code == 0 else ".failed"))
    print(f"[{time.strftime('%H:%M:%S')}] {job.name}: код {code}", flush=True)


def main() -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    print(f"Раннер слушает {QUEUE} (Ctrl+C — выход)", flush=True)
    while True:
        jobs = sorted(QUEUE.glob("*.job"), key=lambda p: p.stat().st_mtime)
        for job in jobs:
            run_job(job)
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
