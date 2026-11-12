# Текущая конфигурация локальной модели команды (12.11.2026)

Так у нас настроен Unsloth Studio на машине, где идут локальные прогоны. Ключ API не храним
в репозитории — он показывается в Studio и кладётся в свой `.env` (см. `.env.local.example`).

| Параметр Studio | Значение |
|---|---|
| Модель | `unsloth/Qwen3.8-27B-GGUF`, квантование `UD-IQ3_XXS` |
| Имя модели в API | `unsloth/Qwen3.8-27B-GGUF` (именно так, иначе 404 «model is downloaded but not loaded») |
| API | `http://127.0.0.1:8888/v1`, ключ вида `sk-unsloth-…` |
| Context Length | 64 896 (рекомендуем 32 768: запросу нужно ~8–9k токенов на слот) |
| Parallel Slots | 2 (`MAX_WORKERS=2` в `.env`) |
| KV Cache Dtype | рекомендуем `q8_0` (по умолчанию Studio ставит `q4_0`) |
| Speculative Decoding | MTP, Draft Tokens 2 |
| Vision | выкл |
| Reasoning Budget | unlimited (размышления включает/выключает наш код) |
| Extra Arguments | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.00` (без `--reasoning-effort`) |
| Эмбеддинги | пока нет (`USE_EMBEDDINGS=0`); план — `ollama pull bge-m3` с выключенным VPN |
| Реранкер | нет (`USE_RERANKER=0`) |

Что улучшило бы точность локальных сравнений: квантование `UD-Q3_K_XL`/`Q4_K_XL` вместо
`IQ3_XXS`, KV `q8_0`, bge-m3 через Ollama.

## Как запускаем прогоны

```
cd energytool-ai
copy .env.local.example .env      # вписать ключ из Studio
python eval\runner.py             # держать окно открытым; задания кладутся в eval\queue\*.job
```

или вручную:

```
python eval\run_eval.py zip  --name v04_82 --zip leaderboard\submission_v04.zip --questions eval\extra_questions.csv
python eval\run_eval.py gen  --name test   --questions eval\extra_questions.csv --vs v04_82
python eval\run_eval.py gen  --name base_82 --questions eval\extra_questions.csv --set PIPELINE=base   # имитация бейзлайна
```

Файлы кейса (`demo_questions.csv`, `demo_output.csv`, `knowledge-base.zip`) лежат в `..\data\`
рядом с репозиторием.
