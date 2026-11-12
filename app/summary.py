"""Резюме заявки для менеджера пресейла и подсказки состояния диалога для LLM."""

from __future__ import annotations

import re
from typing import Sequence

from energy_rag import Consultant, clean_answer, strip_reasoning

SLOT_LABELS = {
    "object_type": "тип объекта",
    "region": "регион",
    "power_kw": "присоединённая мощность, кВт",
    "consumption_kwh": "потребление, кВт·ч/мес",
    "roof": "кровля/площадка",
    "meters_count": "приборов учёта",
    "collection": "сбор показаний сейчас",
    "price_category": "ценовая категория",
    "voltage": "уровень напряжения",
    "resource": "ресурс",
    "has_hourly": "есть почасовые данные",
    "daily_only": "только суточные данные",
    "archive_years": "глубина архива, лет",
    "goals": "цели",
}


def slots_summary(state: dict) -> str:
    slots = state.get("slots") or {}
    parts = []
    for key, label in SLOT_LABELS.items():
        value = slots.get(key)
        if value in (None, "", [], False):
            continue
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        parts.append(f"{label}: {value}")
    return "; ".join(parts)


def history_text(messages: Sequence[dict], limit: int = 16) -> str:
    lines = []
    for message in messages[-limit:]:
        if (message.get("meta") or {}).get("kind") == "lead":
            continue
        role = "Клиент" if message.get("role") == "user" else "Консультант"
        content = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_summary(bot: Consultant | None, state: dict, lead: dict, messages: Sequence[dict], fallback: str) -> str:
    """Резюме заявки через LLM; при недоступности модели — детерминированный шаблон."""
    if bot is None or not bot.llm_ready:
        return fallback
    prompt = f"""Ниже — диалог клиента с ИИ-консультантом МТС EnergyTool и контакты клиента.
Подготовь резюме заявки для менеджера пресейла, чтобы он сразу перешёл к предметному разговору.
Формат — короткие строки «Поле: значение»:
Клиент; Компания; Телефон; Задача клиента; Объект и известные параметры; Что уже обсудили;
Рекомендованное направление EnergyTool; Что уточнить на звонке.
Используй только факты из диалога, ничего не придумывай; если параметр неизвестен — пиши «не указано».

Контакты: ФИО — {lead.get('name')}; компания — {lead.get('company')}; телефон — {lead.get('phone')}
Собранные параметры: {slots_summary(state) or 'не собраны'}

ДИАЛОГ:
{history_text(messages)}"""
    text = bot.complete(
        [
            {"role": "system", "content": "Ты аккуратно готовишь резюме заявок для менеджеров. Не выдумывай данные."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
    )
    if not text:
        return fallback
    text = strip_reasoning(text).replace("**", "").strip()
    return text or clean_answer(fallback)
