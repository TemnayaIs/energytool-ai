"""Локальная оценка ответов по методике LLM-судьи из технического регламента.

Работает с любым OpenAI-совместимым сервером (Unsloth Studio, llama-server, LM Studio)
и с прокси платформы. Настройки — в .env в корне проекта (см. docs/LOCAL_LLM.md).

Команды (запуск из корня проекта):

    # 1. Ответы текущей версии (варианты — через --set КЛЮЧ=ЗНАЧЕНИЕ)
    python eval/run_eval.py gen --name v02
    python eval/run_eval.py gen --name v02_check --set SELF_CHECK=1

    # 2. Ответы старой версии из ZIP (точка отсчёта, на платформе 0.31423)
    python eval/run_eval.py zip --name v01 --zip leaderboard/submission_0.31423.zip

    # 3. Судья: B против A с опорой на эталон, два раунда с перестановкой
    python eval/run_eval.py judge --a v01 --b v02

    # Всё сразу: gen + judge против базовой версии
    python eval/run_eval.py gen --name v02_check --set SELF_CHECK=1 --vs v01

Результаты пишутся в eval/runs/ (папка в .gitignore).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
DEFAULT_QUESTIONS = [ROOT.parent / "data" / "demo_questions.csv", ROOT / "eval" / "extra_questions.csv"]
DEFAULT_GOLDEN = [ROOT.parent / "data" / "demo_output.csv", ROOT / "eval" / "extra_golden.csv"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Мини-загрузчик .env без зависимостей; переменные окружения имеют приоритет."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):
            os.environ[key] = value


def read_csv(paths: list[Path], column: str) -> dict[str, str]:
    """Читает id -> column из нескольких файлов; id дополнительных наборов с префиксом."""
    out: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        prefix = "" if path.parent.name == "data" else f"{path.stem.split('_')[0]}-"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                out[prefix + str(row["id"]).strip()] = row[column]
    return out


def write_answers(name: str, answers: dict[str, str], meta: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        writer.writerows(answers.items())
    (RUNS / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_answers(name: str) -> dict[str, str]:
    with (RUNS / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row["answer"] for row in csv.DictReader(handle)}


def apply_overrides(pairs: list[str]) -> dict[str, str]:
    overrides = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        os.environ[key.strip()] = value.strip()
        overrides[key.strip()] = value.strip()
    return overrides


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------


def preflight() -> None:
    """Один короткий запрос к модели: ошибки настройки видны сразу, а не после 18 вопросов."""
    sys.path.insert(0, str(ROOT))
    import energy_rag as er

    settings = er.Settings.from_env()
    if not settings.llm_ready:
        raise SystemExit("LLM_BASE_URL/LLM_API_KEY не заданы (.env)")
    client = er.ProxyClient(settings, retries=1)
    try:
        reply = client.chat([{"role": "user", "content": "Ответь одним словом: готов?"}], max_tokens=20)
        print(f"Модель {settings.model}: OK ({reply[:40]!r})")
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"Модель {settings.model} недоступна: {error}")


def cmd_gen(args) -> str:
    # пример с тем же вопросом в промпт не берём, иначе оценка нечестная
    os.environ.setdefault("FEWSHOT_EXCLUDE_EXACT", "1")
    overrides = apply_overrides(args.set)
    if os.environ.get("PIPELINE") == "base":
        return cmd_base(args)
    preflight()
    sys.path.insert(0, str(ROOT))
    import energy_rag as er  # импорт после применения переменных

    questions = read_csv([Path(p) for p in args.questions], "question")
    if args.limit:
        questions = dict(list(questions.items())[: args.limit])
    kb = args.kb or (ROOT / "data" / "knowledge-base")
    bot = er.Consultant(kb, cache_dir=RUNS / "cache")
    print(f"[{args.name}] вопросов: {len(questions)}; поиск: {bot.retriever.mode}; LLM: {bot.llm_ready}; {overrides}")
    if bot.retriever.dense_error:
        print(f"  эмбеддинги недоступны: {bot.retriever.dense_error}")

    answers: dict[str, str] = {}
    lat, modes = [], {}

    def one(item):
        qid, q = item
        res = bot.answer(er.parse_dialog(q))
        return qid, res

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (qid, res) in enumerate(pool.map(one, questions.items()), 1):
            answers[qid] = res.text
            lat.append(res.latency_ms)
            modes[res.mode] = modes.get(res.mode, 0) + 1
            flag = "" if res.mode == "llm" else f"  [{res.mode}: {res.error}]"
            print(f"  {i}/{len(questions)} id={qid} {res.latency_ms} мс, {len(res.text.split())} сл.{flag}")
    meta = {
        "name": args.name,
        "overrides": overrides,
        "retrieval": bot.retriever.mode,
        "modes": modes,
        "seconds": round(time.monotonic() - started, 1),
        "latency_ms_median": statistics.median(lat) if lat else 0,
        "words_mean": round(statistics.mean(len(a.split()) for a in answers.values()), 1),
    }
    path = write_answers(args.name, answers, meta)
    print(f"  -> {path}  {meta}")
    return args.name


BASELINE_PROMPT = """Ты — консультант компании МТС Energytool. Ответь на вопрос пользователя, используя информацию из контекста.
Если ответа нет в контексте, скажи, что информации недостаточно.

Контекст:
{context}

Вопрос: {question}

Ответ:"""


def cmd_base(args) -> str:
    """Имитация простого бейзлайна: BM25 top-5 + «ответь по контексту», без требований к стилю."""
    preflight()
    sys.path.insert(0, str(ROOT))
    import energy_rag as er

    settings = er.Settings.from_env()
    settings.use_embeddings = False
    settings.use_reranker = False
    settings.top_k = 5
    kb = er.KnowledgeBase.load(args.kb or (ROOT / "data" / "knowledge-base"))
    retriever = er.Retriever(kb, None, settings)
    client = er.ProxyClient(settings)
    questions = read_csv([Path(p) for p in args.questions], "question")
    if args.limit:
        questions = dict(list(questions.items())[: args.limit])

    def one(item):
        qid, q = item
        hits = retriever.search(er.parse_dialog(q))[:5]
        context = "\n\n".join(h.chunk.text for h in hits)
        prompt = BASELINE_PROMPT.format(context=context, question=q)
        try:
            text = client.chat([{"role": "user", "content": prompt}], max_tokens=700, temperature=0.3)
        except Exception as error:  # noqa: BLE001
            text = f"Информации недостаточно. ({type(error).__name__})"
        return qid, " ".join(text.split())

    answers: dict[str, str] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (qid, text) in enumerate(pool.map(one, questions.items()), 1):
            answers[qid] = text
            print(f"  {i}/{len(questions)} id={qid} {len(text.split())} сл.")
    meta = {
        "name": args.name,
        "pipeline": "baseline-imitation",
        "seconds": round(time.monotonic() - started, 1),
        "words_mean": round(statistics.mean(len(a.split()) for a in answers.values()), 1),
    }
    print(f"  -> {write_answers(args.name, answers, meta)}  {meta}")
    return args.name


def cmd_zip(args) -> str:
    """Запускает solution.py из ZIP-сабмита на тех же вопросах (как платформа)."""
    apply_overrides(args.set)
    load_dotenv()
    preflight()
    questions = read_csv([Path(p) for p in args.questions], "question")
    if args.limit:
        questions = dict(list(questions.items())[: args.limit])
    kb_zip = Path(args.kb_zip)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        shutil.unpack_archive(args.zip, work)
        with (work / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "question"])
            writer.writerows(questions.items())
        shutil.copy(kb_zip, work / "knowledge-base.zip")
        env = {**os.environ, "DATA_DIR": str(work), "PYTHONPATH": str(work), "PYTHONIOENCODING": "utf-8"}
        started = time.monotonic()
        subprocess.run([sys.executable, "solution.py"], cwd=work, env=env, check=True)
        with (work / "submission.csv").open(encoding="utf-8", newline="") as handle:
            answers = {row["id"]: row["answer"] for row in csv.DictReader(handle)}
    meta = {"name": args.name, "zip": str(args.zip), "seconds": round(time.monotonic() - started, 1)}
    print(f"  -> {write_answers(args.name, answers, meta)}  {meta}")
    return args.name


# ---------------------------------------------------------------------------
# Судья (методика регламента)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """Ты — строгий эксперт, оценивающий ответы ИИ-консультанта МТС EnergyTool (энергетические продукты для бизнеса).
Сравни два ответа на один вопрос клиента, ориентируясь на эталонный ответ.
Критерии по важности: 1) фактическая точность и соответствие эталону (ключевые факты, оговорки, отсутствие выдумок и противоречий);
2) полнота: отвечает ли на сам вопрос, есть ли ключевые факты эталона; 3) полезность консультанта: понятность, уместный уточняющий вопрос,
краткость (длинный ответ без новых полезных фактов — минус). Порядок ответов не должен влиять на оценку."""

JUDGE_TEMPLATE = """ВОПРОС КЛИЕНТА (может содержать историю диалога):
{question}

ЭТАЛОННЫЙ ОТВЕТ:
{golden}

ОТВЕТ A:
{a}

ОТВЕТ B:
{b}

Выбери одно:
1 — A значительно лучше B
2 — A немного лучше B
3 — ответы сопоставимы по качеству
4 — B немного лучше A
5 — B значительно лучше A

Сначала одной-двумя фразами укажи главное различие, затем последней строкой выведи «ОЦЕНКА: <цифра>»."""


def judge_once(client, question: str, golden: str, a: str, b: str) -> int | None:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, golden=golden, a=a, b=b)},
    ]
    for _ in range(2):
        try:
            text = client.chat(messages, max_tokens=300, temperature=0.0)
        except Exception as error:  # noqa: BLE001
            print(f"    судья: {type(error).__name__}: {error}")
            continue
        match = re.search(r"ОЦЕНКА\s*[:：]?\s*([1-5])", text) or re.search(r"([1-5])\s*$", text.strip())
        if match:
            return int(match.group(1))
    return None


def cmd_judge(args) -> dict:
    load_dotenv()
    sys.path.insert(0, str(ROOT))
    import energy_rag as er

    settings = er.Settings.from_env()
    if args.judge_think:
        settings.enable_thinking = True
    client = er.ProxyClient(settings)
    questions = read_csv([Path(p) for p in args.questions], "question")
    golden = read_csv([Path(p) for p in args.golden], "answer")
    base, team = load_answers(args.a), load_answers(args.b)
    ids = [i for i in team if i in base and i in golden and i in questions]
    if args.limit:
        ids = ids[: args.limit]
    print(f"Судья: {args.b} (команда) против {args.a} (baseline), вопросов: {len(ids)}")

    def score(qid):
        # Раунд 1: A — baseline, B — команда; раунд 2: наоборот (как в регламенте).
        r1 = judge_once(client, questions[qid], golden[qid], base[qid], team[qid])
        r2 = judge_once(client, questions[qid], golden[qid], team[qid], base[qid])
        s1 = None if r1 is None else (r1 - 1) / 4
        s2 = None if r2 is None else (5 - r2) / 4
        return qid, s1, s2

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for qid, s1, s2 in pool.map(score, ids):
            if s1 is None or s2 is None:
                print(f"  id={qid}: судья не дал оценку — исключено")
                continue
            s = (s1 + s2) / 2
            rows.append((qid, s1, s2, s))
            mark = "WIN " if s > 0.5 else "LOSS" if s < 0.5 else "tie "
            print(f"  {mark} id={qid}: {s:.3f} (r1={s1:.2f}, r2={s2:.2f})")

    if not rows:
        print("Нет оценённых вопросов")
        return {}
    match_rate = sum(r[3] for r in rows) / len(rows)
    wins = sum(r[3] > 0.5 for r in rows)
    losses = sum(r[3] < 0.5 for r in rows)
    win_rate = wins / (wins + losses) if wins + losses else 0.5
    total = 0.7 * match_rate + 0.3 * win_rate
    bias = statistics.mean(r[1] - r[2] for r in rows)
    result = {
        "team": args.b,
        "baseline": args.a,
        "n": len(rows),
        "MatchRate": round(match_rate, 4),
        "WinRate": round(win_rate, 4),
        "Score": round(total, 4),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "position_bias": round(bias, 3),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / f"judge_{args.b}_vs_{args.a}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "round1", "round2", "score", "question", "golden", "baseline", "team"])
        for qid, s1, s2, s in rows:
            writer.writerow([qid, s1, s2, s, questions[qid], golden[qid], base[qid], team[qid]])
    print(json.dumps(result, ensure_ascii=False))
    with (RUNS / "leaderboard.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**result, "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--questions", nargs="+", default=[str(p_) for p_ in DEFAULT_QUESTIONS])
        p.add_argument("--limit", type=int, default=0, help="только первые N вопросов")
        p.add_argument("--workers", type=int, default=int(os.getenv("MAX_WORKERS", "2")))

    g = sub.add_parser("gen", help="ответы текущей версии")
    common(g)
    g.add_argument("--name", required=True)
    g.add_argument("--set", nargs="*", default=[], help="переменные окружения варианта, КЛЮЧ=ЗНАЧЕНИЕ")
    g.add_argument("--kb", default="")
    g.add_argument("--vs", default="", help="сразу сравнить с этим прогоном")
    g.add_argument("--golden", nargs="+", default=[str(p) for p in DEFAULT_GOLDEN])
    g.add_argument("--judge-think", action="store_true")

    z = sub.add_parser("zip", help="ответы из ZIP-сабмита")
    common(z)
    z.add_argument("--name", required=True)
    z.add_argument("--zip", required=True)
    z.add_argument("--kb-zip", default=str(ROOT.parent / "data" / "knowledge-base.zip"))
    z.add_argument("--set", nargs="*", default=[], help="переменные окружения прогона, КЛЮЧ=ЗНАЧЕНИЕ")

    bsp = sub.add_parser("base", help="ответы имитации бейзлайна организаторов")
    common(bsp)
    bsp.add_argument("--name", required=True)
    bsp.add_argument("--kb", default="")

    j = sub.add_parser("judge", help="сравнение двух прогонов")
    common(j)
    j.add_argument("--a", required=True, help="baseline")
    j.add_argument("--b", required=True, help="команда")
    j.add_argument("--golden", nargs="+", default=[str(p) for p in DEFAULT_GOLDEN])
    j.add_argument("--judge-think", action="store_true", help="судья с размышлениями (медленнее, точнее)")

    args = parser.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
        if args.vs:
            args.a, args.b = args.vs, args.name
            for key in ("SELF_CHECK", "ENABLE_THINKING"):  # судья всегда в одном режиме
                os.environ.pop(key, None)
            cmd_judge(args)
    elif args.cmd == "zip":
        cmd_zip(args)
    elif args.cmd == "base":
        cmd_base(args)
    else:
        cmd_judge(args)


if __name__ == "__main__":
    main()
