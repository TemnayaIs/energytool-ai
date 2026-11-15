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

from energy_rag import Consultant, dialog_from_messages

from . import config, consultant, store
from .summary import build_summary
from .schemas import ChatIn, LeadIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("energytool")

BOT: Consultant | None = None
STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BOT
    store.init_db()
    BOT = Consultant(config.KB_PATH, cache_dir=config.DB_PATH.parent)
    logger.info("База знаний: %s фрагментов из %s; поиск: %s", len(BOT.kb.chunks), config.KB_PATH, BOT.retriever.mode)
    if BOT.retriever.dense_error:
        logger.warning("Эмбеддинги недоступны, работаем на BM25: %s", BOT.retriever.dense_error)
    logger.info("Режим ответов: %s", "LLM " + BOT.s.model if BOT.llm_ready else "локальный демо-режим")
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
        "kb_chunks": len(BOT.kb.chunks) if BOT else 0,
        "llm_enabled": bool(BOT and BOT.llm_ready),
        "llm_model": BOT.s.model if BOT and BOT.llm_ready else None,
        "retrieval": BOT.retriever.mode if BOT else None,
        "reranker_enabled": bool(BOT and "rerank" in BOT.retriever.mode),
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


@app.post("/api/sessions/clear")
async def clear_sessions():
    count = store.clear_all_sessions()
    return {"cleared": count}


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: str):
    bundle = store.get_session_bundle(session_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return bundle


@app.post("/api/chat")
async def chat(payload: ChatIn):
    if BOT is None:
        raise HTTPException(status_code=503, detail="База знаний ещё загружается")

    session = store.get_session(payload.session_id) if payload.session_id else None
    if not session:
        session = store.create_session()

    state = dict(consultant.new_state())
    state.update(session.get("state") or {})
    history = store.get_messages(session["id"])

    plan = consultant.plan_turn(payload.message, state)
    dialog = dialog_from_messages(history, payload.message)

    result = BOT.answer(dialog, extra=consultant.llm_hint(plan, state))
    answer, mode = result.text, result.mode
    started = time.perf_counter()
    if mode != "llm":
        # Демо-режим без ключей (или сбой LLM): курируемые ответы + факты из базы.
        local_results = [(hit.chunk, hit.score) for hit in result.hits]
        answer = consultant.build_local_answer(payload.message, state, local_results, force_lead=plan.offer_lead)
        mode = "local" if not BOT.llm_ready else "local-fallback"

    if plan.offer_lead and not re.search(r"заявк|форм|контакт", answer, re.IGNORECASE):
        answer = answer.rstrip() + "\n\nЕсли удобно, оставьте ФИО, телефон и название компании — передам заявку специалисту, он подготовит бесплатный расчёт."

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    latency_ms = result.latency_ms + round((time.perf_counter() - started) * 1000)
    sources = result.sources(limit=6)

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
            "sources": sources,
        },
    )
    store.save_state(session["id"], state, title=title)

    return {
        "session_id": session["id"],
        "title": title,
        "answer": answer,
        "sources": sources,
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
            "llm_model": BOT.s.model if mode == "llm" else None,
            "retrieval": BOT.retriever.mode,
            "error": result.error or None,
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
    summary = build_summary(BOT, state, lead, messages, fallback)

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
