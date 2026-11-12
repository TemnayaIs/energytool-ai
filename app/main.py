"""FastAPI-приложение ИИ-консультанта МТС EnergyTool."""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_core import BM25Index, load_chunks, search, sources_payload

from . import config, consultant, llm, store
from .schemas import ChatIn, LeadIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("energytool")

KB_INDEX: BM25Index | None = None
KB_CHUNKS = 0
STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global KB_INDEX, KB_CHUNKS
    store.init_db()
    chunks = load_chunks(config.KB_PATH)
    KB_INDEX = BM25Index(chunks)
    KB_CHUNKS = len(chunks)
    logger.info("База знаний загружена: %s фрагментов из %s", KB_CHUNKS, config.KB_PATH)
    logger.info("Режим ответов: %s", "LLM " + config.LLM_MODEL if llm.llm_available() else "локальный демо-режим")
    yield


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index_page():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "kb_chunks": KB_CHUNKS,
        "llm_enabled": llm.llm_available(),
        "llm_model": config.LLM_MODEL if llm.llm_available() else None,
        "reranker_enabled": config.USE_RERANKER,
        "started_at": STARTED_AT.isoformat(),
        "stats": store.stats(),
    }


@app.post("/api/session/new")
async def new_session():
    session = store.create_session()
    return {"session_id": session["id"], "created_at": session["created_at"]}


@app.get("/api/sessions")
async def sessions():
    return {"sessions": store.list_sessions(limit=25)}


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: str):
    bundle = store.get_session_bundle(session_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return bundle


@app.post("/api/chat")
async def chat(payload: ChatIn):
    if KB_INDEX is None:
        raise HTTPException(status_code=503, detail="База знаний ещё загружается")

    session = store.get_session(payload.session_id) if payload.session_id else None
    if not session:
        session = store.create_session()

    state = dict(consultant.new_state())
    state.update(session.get("state") or {})
    history = store.get_messages(session["id"])

    plan = consultant.plan_turn(payload.message, state)

    started = time.perf_counter()
    results = search(KB_INDEX, plan.context, intent=plan.intent, top_k=config.TOP_K)
    results = llm.rerank(plan.query, results)

    answer: str | None = None
    mode = "local"
    if llm.llm_available():
        answer = llm.generate_answer(
            payload.message,
            state,
            history,
            results,
            offer_lead=plan.offer_lead,
        )
        mode = "llm" if answer else "local-fallback"

    if not answer:
        answer = consultant.build_local_answer(
            payload.message,
            state,
            results,
            force_lead=plan.offer_lead,
        )

    if plan.offer_lead and "заявк" not in answer.lower() and "форм" not in answer.lower():
        answer = answer.rstrip() + "\n\nЗаполните короткую форму — передам заявку инженеру, он подготовит бесплатный расчёт."

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    latency_ms = round((time.perf_counter() - started) * 1000)

    if plan.question:
        asked = list(state.get("asked") or [])
        asked.append(plan.question)
        state["asked"] = asked[-8:]
    state["stage"] = "lead_offered" if plan.offer_lead else ("consult" if state.get("turn", 0) > 1 else "discovery")

    title = session.get("title") or "Новый диалог"
    if title == "Новый диалог":
        # Во встроенных диалогах берём последнюю реплику клиента, а не весь лог.
        lines = [line.strip() for line in payload.message.splitlines() if line.strip()]
        last_user = ""
        for line in lines:
            lowered = line.lower()
            if lowered.startswith(("пользователь:", "user:")):
                last_user = line.split(":", 1)[1].strip()
        title = re.sub(r"\s+", " ", last_user or (lines[-1] if lines else payload.message))[:72]

    store.add_message(
        session["id"],
        "user",
        payload.message,
        {"intent": plan.intent},
    )
    store.add_message(
        session["id"],
        "assistant",
        answer,
        {
            "mode": mode,
            "latency_ms": latency_ms,
            "intent": plan.intent,
            "sources": sources_payload(results, limit=5),
        },
    )
    store.save_state(session["id"], state, title=title)

    return {
        "session_id": session["id"],
        "title": title,
        "answer": answer,
        "sources": sources_payload(results, limit=5),
        "quick_replies": plan.quick_replies,
        "lead": {
            "offered": plan.offer_lead,
            "done": bool(state.get("lead_done")),
            "prefill": {
                "name": state.get("slots", {}).get("name", ""),
                "phone": state.get("slots", {}).get("phone", ""),
                "company": state.get("slots", {}).get("company", ""),
            },
        },
        "debug": {
            "latency_ms": latency_ms,
            "mode": mode,
            "intent": plan.intent,
            "intent_label": consultant.INTENT_LABELS.get(plan.intent, plan.intent),
            "product": plan.product,
            "turn": state.get("turn", 0),
            "slots": plan.slots,
            "llm_model": config.LLM_MODEL if mode == "llm" else None,
        },
    }


@app.post("/api/lead")
async def create_lead(payload: LeadIn):
    session = store.get_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Диалог не найден")

    state = dict(consultant.new_state())
    state.update(session.get("state") or {})
    messages = store.get_messages(payload.session_id)

    lead = {"name": payload.name, "phone": payload.phone, "company": payload.company}
    fallback = consultant.make_summary(state, lead, messages)
    summary = llm.generate_summary(state, lead, messages, fallback)

    lead_id = store.add_lead(payload.session_id, payload.name, payload.phone, payload.company, summary)

    state["lead_done"] = True
    state["stage"] = "lead_done"
    state["slots"] = {**state.get("slots", {}), **lead}
    store.save_state(payload.session_id, state)

    confirmation = (
        f"Заявка №{lead_id} сформирована. Передал её менеджеру: он свяжется с вами, "
        "уточнит недостающие параметры и подготовит бесплатный расчёт."
    )
    store.add_message(payload.session_id, "assistant", confirmation, {"kind": "lead", "lead_id": lead_id})

    return {
        "ok": True,
        "lead_id": lead_id,
        "summary": summary,
        "confirmation": confirmation,
    }
