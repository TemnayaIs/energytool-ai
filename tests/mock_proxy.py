"""Mock OpenAI-совместимого прокси для локальных тестов без ключей.

Запуск: python -m uvicorn tests.mock_proxy:app --port 8911
MOCK_REJECT_KWARGS=1 — имитировать прокси, который не принимает chat_template_kwargs.
"""
import hashlib, os, random, math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
app = FastAPI()
calls = {"chat": 0, "emb": 0, "rerank": 0}
REJECT = os.getenv("MOCK_REJECT_KWARGS") == "1"
def vec(t):
    h = hashlib.sha256(t.encode()).digest()
    rnd = random.Random(h); return [rnd.uniform(-1, 1) for _ in range(64)]
@app.post("/v1/embeddings")
async def emb(r: Request):
    j = await r.json(); calls["emb"] += 1
    inp = j["input"] if isinstance(j["input"], list) else [j["input"]]
    return {"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(inp)]}
@app.post("/v1/rerank")
async def rerank(r: Request):
    j = await r.json(); calls["rerank"] += 1
    docs = j["documents"]; q = set(j["query"].lower().split())
    sc = [(i, len(q & set(d.lower().split())) / 10) for i, d in enumerate(docs)]
    sc.sort(key=lambda x: -x[1])
    return {"results": [{"index": i, "relevance_score": s} for i, s in sc[: j.get("top_n", len(docs))]]}
@app.post("/v1/chat/completions")
async def chat(r: Request):
    j = await r.json(); calls["chat"] += 1
    if REJECT and "chat_template_kwargs" in j:
        return JSONResponse({"error": "unknown field chat_template_kwargs"}, 400)
    if calls["chat"] % 7 == 0:
        return JSONResponse({"error": "rate"}, 429)
    user = j["messages"][-1]["content"]
    if os.getenv("MOCK_THINK_EMPTY") == "1" and j.get("max_tokens", 0) < 5000:
        return {"choices": [{"message": {"role": "assistant", "content": "<think>долго думаю"}, "finish_reason": "length"}]}
    if "ОЦЕНКА: <цифра>" in user:
        return {"choices": [{"message": {"role": "assistant", "content": f"Различие в полноте.\nОЦЕНКА: {random.randint(2, 5)}"}}]}
    q = user.split("ВОПРОС КЛИЕНТА:\n")[-1].split("ПОСЛЕДНЯЯ РЕПЛИКА КЛИЕНТА (ответь на неё с учётом истории):\n")[-1].split("\n")[0]
    content = f"<think>hmm</think>**Да**, по вопросу «{q[:60]}» можно.\n- пункт один\n- пункт два\nКакой у вас регион?"
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}
@app.get("/stats")
def stats(): return calls
