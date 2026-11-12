# Локальный запуск Qwen3.8-27B и оценка ответов

Платформа хакатона даёт 5 сабмитов в день. Чтобы сравнивать варианты без сабмитов, команда
запускает ту же модель локально и прогоняет локального LLM-судью (методика регламента).
Локальная модель квантованная, поэтому абсолютные числа отличаются от лидерборда —
сравнивайте варианты между собой.

## 1. Модель генерации (Unsloth Studio)

Модель: `unsloth/Qwen3.8-27B-GGUF`, квантование `UD-IQ3_XXS` (если хватает VRAM — лучше
`UD-Q3_K_XL` или `Q4_K_M`).

| Параметр | Значение | Комментарий |
|---|---|---|
| Context Length | 32768 | 2 слота × 16k; запрос консультанта ≈ 8–9k токенов |
| KV Cache Dtype | q8_0 | q4_0 портит качество на длинном контексте |
| Vision | выкл | не нужен, экономит VRAM |
| Speculative Decoding | MTP, Draft Tokens 2 | ускорение без изменения ответов |
| Parallel Slots | 2 | при нехватке VRAM — 1 (и `MAX_WORKERS=1`) |
| Reasoning Budget | unlimited | размышления включает/выключает код (`ENABLE_THINKING`) |
| Chat Template | Default | |
| VRAM Budget | 97% (90%, если bge-m3 на той же GPU) | |
| Extra Arguments | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.00` | без `--reasoning-effort`; температуру код передаёт сам (0.1) |

API Studio: `http://127.0.0.1:8888/v1`, ключ показывается в Studio.

Аналог для голого llama.cpp:

```
llama-server -m Qwen3.8-27B-UD-Q3_K_XL.gguf --host 127.0.0.1 --port 8888 --jinja -c 32768 --parallel 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0
```

## 2. Эмбеддинги bge-m3 (необязательно)

```
ollama pull bge-m3          # поднимется на http://127.0.0.1:11434/v1
```

Если `ollama pull` падает с `resolves to non-public 198.18.x.x` — мешает VPN/прокси в режиме
TUN/fake-IP: выключите его на время скачивания. Без bge-m3 поиск работает на BM25
(`USE_EMBEDDINGS=0`), это хуже, чем на платформе.

## 3. Файл .env (в git не попадает)

```env
LLM_BASE_URL=http://127.0.0.1:8888/v1
LLM_API_KEY=<ключ из Unsloth Studio>
LLM_MODEL=unsloth/Qwen3.8-27B-GGUF
LLM_TIMEOUT=180
MAX_WORKERS=2
USE_RERANKER=0
USE_EMBEDDINGS=0
# после ollama pull bge-m3:
# USE_EMBEDDINGS=1
# EMB_BASE_URL=http://127.0.0.1:11434/v1
# EMB_API_KEY=ollama
# EMB_MODEL=bge-m3
```

## 4. Оценка

Нужен Python 3.11+ и `pip install httpx==0.28.1`. Файлы кейса `demo_questions.csv`,
`demo_output.csv`, `knowledge-base.zip` лежат в папке `data/` рядом с репозиторием
(`../data/`). Запуск из корня репозитория:

```
python eval/run_eval.py zip  --name v01 --zip leaderboard/submission_0.31423.zip   # старая версия (0.314 на платформе)
python eval/run_eval.py gen  --name v02 --vs v01                                   # текущая версия + судья
python eval/run_eval.py gen  --name v02_check --set SELF_CHECK=1 --vs v01          # вариант с самопроверкой
python eval/run_eval.py gen  --name v02_think --set ENABLE_THINKING=1 --vs v01     # вариант с размышлениями
python eval/run_eval.py judge --a v02 --b v02_check                                # сравнить два варианта напрямую
```

Судья выдаёт `Score = 0.7·MatchRate + 0.3·WinRate` (0,5 — паритет). Каждый вопрос оценивается
дважды с перестановкой ответов. Подробности по вопросам — `eval/runs/judge_*.csv`, история —
`eval/runs/leaderboard.jsonl`. `position_bias` заметно больше 0 — судья предпочитает позицию,
доверяйте только устойчивым различиям.

Переключатели вариантов (`--set`): `SELF_CHECK=1`, `ENABLE_THINKING=1`, `TOP_K=12`,
`USE_EMBEDDINGS=0/1`, `TEMPERATURE=0.3`.

## 5. Очередь прогонов (раннер)

`python eval/runner.py` — держите окно открытым. Раннер выполняет задания из `eval/queue/*.job`
(одна строка аргументов `run_eval.py`, например `gen --name v03 --vs v02`), пишет лог в
`eval/queue/<имя>.log`. Так прогоны можно ставить в очередь, не занимая терминал.
Разрешены только подкоманды `gen`, `zip`, `judge`.
