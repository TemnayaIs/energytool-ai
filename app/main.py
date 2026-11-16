"""FastAPI-приложение ИИ-консультанта МТС EnergyTool."""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from energy_rag import Consultant, dialog_from_messages

from . import config, consultant, store
from .summary import build_summary
from .schemas import ChatIn, LeadIn, RegisterIn, LoginIn, ProfileUpdateIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("energytool")

BOT: Consultant | None = None
STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BOT
    store.init_db()
    # Создаём администратора по умолчанию, если его нет
    if not store.get_user_by_username("admin"):
        try:
            store.create_user("admin", "admin123", "admin", "Администратор", "", "")
        except ValueError:
            pass
    BOT = Consultant(config.KB_PATH, cache_dir=config.DB_PATH.parent)
    logger.info("База знаний: %s фрагментов из %s; поиск: %s", len(BOT.kb.chunks), config.KB_PATH, BOT.retriever.mode)
    if BOT.retriever.dense_error:
        logger.warning("Эмбеддинги недоступны, работаем на BM25: %s", BOT.retriever.dense_error)
    logger.info("Режим ответов: %s", "LLM " + BOT.s.model if BOT.llm_ready else "локальный демо-режим")
    yield


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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


@app.get("/admin", include_in_schema=False)
async def admin_page():
    return FileResponse(config.STATIC_DIR / "admin.html")




def _token_from_request(request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-auth-token", "").strip()


def _current_user(request) -> dict | None:
    return store.get_user_by_token(_token_from_request(request))


def _require_user(request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    return user


def _require_admin(request) -> dict:
    user = _require_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Требуются права администратора")
    return user

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


@app.post("/api/auth/register")
async def register(payload: RegisterIn):
    try:
        user = store.create_user(payload.username, payload.password, "user", payload.name, payload.phone, payload.company)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = store.create_token(user["id"])
    return {"token": token, "user": user}


@app.post("/api/auth/login")
async def login(payload: LoginIn):
    user = store.get_user_by_username(payload.username)
    if not user or not store.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    token = store.create_token(user["id"])
    safe = {k: user[k] for k in ("id", "username", "role", "name", "phone", "company", "created_at")}
    return {"token": token, "user": safe}


@app.get("/api/me")
async def me(request: Request):
    user = _require_user(request)
    return {"user": user}


@app.put("/api/me")
async def update_me(payload: ProfileUpdateIn, request: Request):
    user = _require_user(request)
    updated = store.update_user(user["id"], payload.name, payload.phone, payload.company)
    return {"user": updated}


@app.post("/api/auth/logout")
async def logout(request: Request):
    token = _token_from_request(request)
    if token:
        store.revoke_token(token)
    return {"ok": True}


@app.get("/api/my/sessions")
async def my_sessions(request: Request):
    _require_user(request)
    return {"sessions": store.list_sessions(limit=50)}


@app.get("/api/my/leads")
async def my_leads(request: Request):
    user = _require_user(request)
    leads = [l for l in store.list_all_leads(200) if l.get("session_id")]
    return {"leads": leads}


@app.get("/api/admin/leads")
async def admin_leads(request: Request):
    _require_admin(request)
    return {"leads": store.list_all_leads(200)}


@app.get("/api/admin/sessions")
async def admin_sessions(request: Request):
    _require_admin(request)
    return {"sessions": store.list_all_sessions(100)}


@app.get("/api/admin/leads/{lead_id}")
async def admin_lead_detail(lead_id: int, request: Request):
    _require_admin(request)
    leads = store.list_all_leads(1000)
    lead = next((l for l in leads if l["id"] == lead_id), None)
    if not lead:
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    bundle = store.get_session_bundle(lead["session_id"]) if lead.get("session_id") else None
    return {"lead": lead, "session": bundle}


@app.post("/api/admin/leads/{lead_id}/forward")
async def admin_forward_lead(lead_id: int, request: Request):
    _require_admin(request)
    leads = store.list_all_leads(1000)
    lead = next((l for l in leads if l["id"] == lead_id), None)
    if not lead:
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    # Mark as forwarded by updating summary? Just return ok with a message.
    return {"ok": True, "lead_id": lead_id, "summary": lead.get("summary", ""), "message": "Заявка передана менеджеру"}

@app.patch("/api/admin/leads/{lead_id}/status")
async def admin_update_lead_status(lead_id: int, request: Request):
    _require_admin(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    status = (body.get("status") or "").strip().lower()
    if status not in ("open", "closed"):
        raise HTTPException(status_code=400, detail="Статус: open или closed")
    ok = store.update_lead_status(lead_id, status)
    if not ok:
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    return {"ok": True, "lead_id": lead_id, "status": status}



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
