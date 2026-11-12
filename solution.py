"""Сабмит: DATA_DIR/questions.csv + knowledge-base.zip -> submission.csv"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from energy_rag import NO_INFO, Consultant, env, env_number, parse_dialog

STARTED = time.monotonic()

# у платформы лимит 20 минут, оставляем запас
TIME_BUDGET = env_number("TIME_BUDGET_SEC", 1080, 120, 1150)
DEADLINE = STARTED + TIME_BUDGET
MAX_WORKERS = env_number("MAX_WORKERS", 8, 1, 32)


def log(message: str) -> None:
    print(f"[{time.monotonic() - STARTED:6.1f}s] {message}", file=sys.stderr, flush=True)


def find_file(*names: str) -> Path:
    data_dir = Path(env("DATA_DIR", default="data"))
    here = Path(__file__).resolve().parent
    for root in (data_dir, here / data_dir, Path.cwd(), here, Path("/data")):
        for name in names:
            path = root / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Не найдены файлы: {', '.join(names)}")


def read_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            qid = str(row.get("id", "")).strip()
            if qid:
                rows.append((qid, str(row.get("question") or "").strip()))
    return rows


def write_submission(rows: list[tuple[str, str]], path: Path = Path("submission.csv")) -> None:
    seen: set[str] = set()
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid, answer in rows:
            if qid in seen:
                continue
            seen.add(qid)
            answer = (answer or "").strip() or NO_INFO
            writer.writerow([qid, answer[:9900]])
    tmp.replace(path)


def main() -> None:
    questions = read_questions(find_file("questions.csv", "test.csv", "demo_questions.csv"))
    if not questions:
        raise SystemExit("questions.csv пуст")
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "knowledge-base", "kb.zip")
    log(f"вопросов: {len(questions)}; база: {kb_path}")

    answers: dict[str, str] = {qid: NO_INFO for qid, _ in questions}
    # пишем заглушку сразу, чтобы файл был даже при падении
    write_submission([(qid, answers[qid]) for qid, _ in questions])

    bot = Consultant(kb_path, cache_dir="/tmp/energytool-cache")
    log(
        f"фрагментов: {len(bot.kb.chunks)}; поиск: {bot.retriever.mode}; "
        f"LLM: {'да' if bot.llm_ready else 'нет — экстрактивный режим'}"
        + (f"; эмбеддинги недоступны ({bot.retriever.dense_error})" if bot.retriever.dense_error else "")
    )

    stats: dict[str, int] = {}
    progress = {"done": 0, "fast": False}
    work_started = time.monotonic()

    def behind_schedule() -> bool:
        # если при текущем темпе не успеваем, отключаем эмбеддинги запроса и реранкер
        done = progress["done"]
        if done < 4:
            return False
        per_question = (time.monotonic() - work_started) / done
        left = len(questions) - done
        return per_question * left > (DEADLINE - time.monotonic()) * 0.8

    def work(item: tuple[str, str]) -> tuple[str, str]:
        qid, question = item
        if not progress["fast"] and behind_schedule():
            progress["fast"] = True
            log("не успеваем по времени: дальше без реранкера и эмбеддингов запроса")
        result = bot.answer(parse_dialog(question or "Здравствуйте"), deadline=DEADLINE, fast=progress["fast"])
        progress["done"] += 1
        stats[result.mode] = stats.get(result.mode, 0) + 1
        if result.error:
            log(f"id={qid}: {result.mode} ({result.error})")
        return qid, result.text

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, item) for item in questions]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                qid, text = future.result()
                answers[qid] = text
            except Exception as error:
                log(f"ошибка: {type(error).__name__}: {error}")
            if done % 20 == 0:
                log(f"готово {done}/{len(questions)}")
                write_submission([(qid, answers[qid]) for qid, _ in questions])

    write_submission([(qid, answers[qid]) for qid, _ in questions])
    log(f"submission.csv записан: {len(questions)} ответов; режимы: {stats}")


if __name__ == "__main__":
    main()
