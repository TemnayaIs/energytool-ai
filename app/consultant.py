"""Логика консультанта: интенты, слоты, уточняющие вопросы и заявка.

Модуль не делает сетевых вызовов. Он готовит «план хода» для LLM и умеет
самостоятельно собрать ответ из найденных фрагментов базы знаний, если
внешняя модель недоступна. Так прототип работает в демо-режиме без ключей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from answers_core import (
    conversation_context,
    curated_answer,
    current_question,
    detect_intent,
    pick_facts,
    sentences,
)
from rag_core import Chunk, normalize

# Справочники

INTENT_LABELS = {
    "solar": "Солнечная электростанция",
    "accounting": "Облачная платформа учёта",
    "losses": "Поиск коммерческих потерь",
    "reading": "Распознавание показаний по фото",
    "orem": "Выход на оптовый рынок",
    "general": "Общий вопрос об EnergyTool",
    "unknown": "Задача пока не определена",
}

INTENT_PRODUCTS = {
    "solar": "Солнечная электростанция под ключ",
    "accounting": "Облачная платформа учёта электроэнергии",
    "losses": "Поиск коммерческих потерь",
    "reading": "Распознавание показаний по фотографии",
    "orem": "Сопровождение выхода на оптовый рынок",
    "general": "Консультация по продуктам МТС EnergyTool",
}

REGIONS = (
    "москва", "московск", "санкт-петербург", "петербург", "ленинградск", "краснодар",
    "ростов", "астрахан", "ставропол", "кавказ", "татарстан", "казан", "башкортостан",
    "уфа", "свердловск", "екатеринбург", "новосибирск", "тюмен", "челябинск", "самар",
    "нижегородск", "воронеж", "волгоград", "пермск", "красноярск", "иркутск", "омск",
    "томск", "кемеров", "алтай", "приморск", "владивосток", "хабаровск", "камчатк",
    "магадан", "сахалин", "чукотк", "крым", "севастопол", "калининград", "архангельск",
    "мурманск", "ярослав", "тверск", "тульск", "калужск", "брянск", "курск", "белгород",
    "липецк", "тамбов", "пенз", "саратов", "ульяновск", "оренбург", "киров", "удмурт",
    "чуваш", "мордов", "марий", "якут", "бурят", "тыва", "хакас", "дагестан", "чечн",
)

REGION_LABELS = {
    "москва": "Москва",
    "московск": "Московская область",
    "санкт-петербург": "Санкт-Петербург",
    "петербург": "Санкт-Петербург",
    "ленинградск": "Ленинградская область",
    "краснодар": "Краснодарский край",
    "ростов": "Ростовская область",
    "астрахан": "Астраханская область",
    "ставропол": "Ставропольский край",
    "кавказ": "Северный Кавказ",
    "татарстан": "Татарстан",
    "казан": "Казань",
    "башкортостан": "Башкортостан",
    "уфа": "Уфа",
    "свердловск": "Свердловская область",
    "екатеринбург": "Екатеринбург",
    "новосибирск": "Новосибирская область",
    "тюмен": "Тюменская область",
    "челябинск": "Челябинская область",
    "самар": "Самарская область",
    "нижегородск": "Нижегородская область",
    "воронеж": "Воронежская область",
    "волгоград": "Волгоградская область",
    "пермск": "Пермский край",
    "красноярск": "Красноярский край",
    "иркутск": "Иркутская область",
    "омск": "Омская область",
    "томск": "Томская область",
    "кемеров": "Кемеровская область",
    "алтай": "Алтайский край",
    "приморск": "Приморский край",
    "владивосток": "Владивосток",
    "хабаровск": "Хабаровский край",
    "камчатк": "Камчатский край",
    "магадан": "Магаданская область",
    "сахалин": "Сахалинская область",
    "чукотк": "Чукотский автономный округ",
    "крым": "Крым",
    "севастопол": "Севастополь",
    "калининград": "Калининградская область",
    "архангельск": "Архангельская область",
    "мурманск": "Мурманская область",
    "ярослав": "Ярославская область",
    "тверск": "Тверская область",
    "тульск": "Тульская область",
    "калужск": "Калужская область",
    "брянск": "Брянская область",
    "курск": "Курская область",
    "белгород": "Белгородская область",
    "липецк": "Липецкая область",
    "тамбов": "Тамбовская область",
    "пенз": "Пензенская область",
    "саратов": "Саратовская область",
    "ульяновск": "Ульяновская область",
    "оренбург": "Оренбургская область",
    "киров": "Кировская область",
    "удмурт": "Удмуртия",
    "чуваш": "Чувашия",
    "мордов": "Мордовия",
    "марий": "Марий Эл",
    "якут": "Якутия",
    "бурят": "Бурятия",
    "тыва": "Тыва",
    "хакас": "Хакасия",
    "дагестан": "Дагестан",
    "чечн": "Чечня",
}

NUMERALS = {
    "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "пятнадцать": 15, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "сорока": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200,
    "триста": 300, "четыреста": 400, "пятьсот": 500,
}
NUMERAL_RE = re.compile(
    r"\b(" + "|".join(sorted(NUMERALS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

PHONE_RE = re.compile(r"(?:\+7|8)?[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-()]*\d{2}[\s\-()]*\d{2}")
POWER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(квт|kw|киловатт)\b(?!\s*(?:ч|час))", re.IGNORECASE)
MEGAWATT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(мвт|mw|мегаватт)\b", re.IGNORECASE)
CONSUMPTION_RE = re.compile(r"(\d[\d\s.,]*)\s*(квт\s*[·.\-]?\s*ч|квтч|киловатт-час)", re.IGNORECASE)
METERS_RE = re.compile(r"(\d+)\s*(?:счетчик|счётчик|прибор|шт|точк|объект)", re.IGNORECASE)

ROOF_WORDS = {
    "скат": "скатная кровля",
    "двускат": "скатная кровля",
    "плоск": "плоская кровля",
    "шифер": "шиферная кровля",
    "металлочерепиц": "металлочерепица",
    "мягк": "мягкая кровля",
    "земл": "земельный участок",
    "фасад": "фасад здания",
    "навес": "навес",
}

OBJECT_WORDS = {
    "завод": "производство",
    "производств": "производство",
    "склад": "склад",
    "логистическ": "логистический склад",
    "магазин": "ретейл",
    "ретейл": "ретейл",
    "торгов": "торговый объект",
    "бизнес-центр": "бизнес-центр",
    "бц": "бизнес-центр",
    "тц": "торговый центр",
    "управляющ": "управляющая компания",
    "тсж": "ТСЖ",
    "снт": "СНТ",
    "товариществ": "товарищество",
    "сетев": "электросетевая компания",
    "электросетев": "электросетевая компания",
    "энергосбыт": "энергосбытовая компания",
    "ресурсоснабж": "ресурсоснабжающая организация",
    "агро": "агропромышленный объект",
    "гостиниц": "гостиница",
    "офис": "офис",
}

GOAL_WORDS = {
    "снизить": "снизить платёж",
    "сократить": "снизить платёж",
    "экономи": "снизить платёж",
    "дешев": "снизить платёж",
    "надежн": "повысить надёжность",
    "резерв": "повысить надёжность",
    "отключен": "повысить надёжность",
    "автоном": "повысить надёжность",
    "учет": "автоматизировать учёт",
    "учёт": "автоматизировать учёт",
    "потер": "найти потери",
    "хищен": "найти потери",
    "опт": "выйти на оптовый рынок",
    "распозна": "распознавать показания",
    "фото": "распознавать показания",
    "тариф": "разобраться с тарифом",
    "категор": "разобраться с тарифом",
}

OUT_OF_SCOPE_MARKERS = (
    "152-фз", "юридическ", "суд", "ответственност",
    "настроить прибор", "настройка прибор", "прошивк", "схема подключения",
    "техническая документация", "паспорт прибора", "конкурент", "другой поставщик",
)

LEAD_REQUEST_MARKERS = (
    "оставить заявк", "оставьте заявк", "связаться", "перезвон", "менеджер",
    "специалист", "консультац", "позвоните", "хочу заявку", "передайте",
)

GREETING_MARKERS = ("привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро", "hello", "hi")
THANKS_MARKERS = ("спасибо", "благодар", "понятно", "ясно", "хорошо")


def new_state() -> dict:
    return {
        "intent": "unknown",
        "product": "",
        "slots": {},
        "turn": 0,
        "stage": "discovery",
        "lead_offered": False,
        "lead_done": False,
        "topics": [],
        "asked": [],
    }


# Разбор сообщения


def extract_slots(text: str, state: dict) -> dict:
    """Достаёт из реплики параметры объекта, не перетирая уже известные."""
    slots = dict(state.get("slots") or {})
    lowered = normalize(text)
    numbers_text = NUMERAL_RE.sub(lambda m: str(NUMERALS[m.group(1).lower()]), text)

    match = MEGAWATT_RE.search(numbers_text)
    if match:
        slots["power_kw"] = round(float(match.group(1).replace(",", ".")) * 1000)
    else:
        match = POWER_RE.search(numbers_text)
        if match:
            slots["power_kw"] = round(float(match.group(1).replace(",", ".")))

    match = CONSUMPTION_RE.search(numbers_text)
    if match:
        raw = re.sub(r"[^\d.,]", "", match.group(1)).replace(",", ".")
        try:
            value = float(raw)
            if "млн" in lowered or "миллион" in lowered:
                value *= 1_000_000
            slots["consumption_kwh"] = int(value)
        except ValueError:
            pass

    match = METERS_RE.search(numbers_text)
    if match:
        slots["meters_count"] = int(match.group(1))

    for word, label in ROOF_WORDS.items():
        if word in lowered:
            slots["roof"] = label
            break

    for word, label in OBJECT_WORDS.items():
        if re.search(rf"\b{re.escape(word)}", lowered):
            slots["object_type"] = label
            break

    for word, label in GOAL_WORDS.items():
        if word in lowered:
            slots.setdefault("goals", [])
            if label not in slots["goals"]:
                slots["goals"].append(label)
            break

    for region in REGIONS:
        if region in lowered:
            slots["region"] = REGION_LABELS.get(region, region.capitalize())
            break

    if any(word in lowered for word in ("почасов", "получасов", "интервальн", "часовая")):
        slots["has_hourly"] = True
    if any(word in lowered for word in ("суточн", "только сутки", "день и ночь", "дневн и ночн")):
        slots["daily_only"] = True
    if re.search(r"(?:за|архив|глубин\w*)\s*(?:два|2)\s*год", lowered) or "два года" in lowered:
        slots["archive_years"] = 2
    elif (
        re.search(r"(?:за|архив|глубин\w*)\s*(?:один|1)\s*год", lowered)
        or "за год" in lowered
        or "годовой архив" in lowered
    ):
        slots.setdefault("archive_years", 1)

    category = re.search(r"(перв|втор|трет|четверт|пят|шест)\w*\s+(?:ценов\w*\s+)?категор", lowered)
    if not category and any(word in lowered for word in ("категор", "счет", "цк")):
        category = re.search(r"\b(перв|втор|трет|четверт|пят|шест)\w*\b", lowered)
    if category:
        mapping = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6}
        slots["price_category"] = mapping[category.group(1)]
    else:
        short = re.search(r"\b(?:цк|категория)\s*([1-6])\b", lowered)
        if short:
            slots["price_category"] = int(short.group(1))

    for marker, value in (("вн", "ВН"), ("сн-1", "СН-1"), ("сн-2", "СН-2"), ("нн", "НН")):
        if re.search(rf"\b{marker}\b", lowered):
            slots["voltage"] = value
            break
    if "высокое напряжение" in lowered:
        slots["voltage"] = "ВН"
    elif "среднее напряжение" in lowered:
        slots["voltage"] = "СН"
    elif "низкое напряжение" in lowered:
        slots["voltage"] = "НН"

    phone = PHONE_RE.search(text)
    if phone:
        slots["phone"] = re.sub(r"\s+", "", phone.group(0))
    if re.search(r"меня зовут\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)", text):
        slots["name"] = re.search(r"меня зовут\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)", text).group(1)
    company = re.search(r"(?:компания|организация|фирма)\s*[«\"']?([А-ЯЁA-Z][\w\-]+(?:\s+[А-ЯЁA-Z][\w\-]+)?)", text)
    if company:
        slots["company"] = company.group(1)

    return slots


def wants_lead(text: str) -> bool:
    lowered = normalize(text)
    return any(marker in lowered for marker in LEAD_REQUEST_MARKERS)


def is_greeting(text: str) -> bool:
    lowered = normalize(text)
    return len(lowered.split()) <= 6 and any(lowered.startswith(marker) for marker in GREETING_MARKERS)


def is_thanks(text: str) -> bool:
    lowered = normalize(text)
    words = lowered.split()
    if len(words) > 8 or "?" in text:
        return False
    if words and words[0] in ("ок", "окей", "ладно"):
        return True
    return any(marker in lowered for marker in THANKS_MARKERS)


def is_out_of_scope(text: str) -> bool:
    lowered = normalize(text)
    return any(marker in lowered for marker in OUT_OF_SCOPE_MARKERS)


# Уточняющие вопросы


def next_question(intent: str, slots: dict) -> str | None:
    goals = slots.get("goals") or []

    if intent == "solar":
        if not goals:
            return "Что для вас важнее: снизить платёж за электроэнергию или иметь резерв при отключениях?"
        if not slots.get("region"):
            return "В каком регионе находится объект?"
        if not slots.get("roof"):
            return "Какая кровля или площадка рассматривается: скатная, плоская, шифер или земля?"
        if not slots.get("power_kw") and not slots.get("consumption_kwh"):
            return "Какая примерно присоединённая мощность или месячное потребление объекта?"
        if not slots.get("object_type"):
            return "Что это за объект: склад, производство, торговый центр или что-то другое?"
        return None

    if intent == "accounting":
        if not slots.get("object_type"):
            return "Что это за организация: производство, сеть магазинов, бизнес-центр, управляющая компания?"
        if not slots.get("meters_count"):
            return "Сколько примерно приборов учёта нужно подключить?"
        if not slots.get("collection"):
            return "Как сейчас собираете показания: вручную, обходами, по фото или уже есть удалённый сбор?"
        if not goals:
            return "Что важнее видеть в кабинете: расход по цехам, по арендаторам или по отдельным объектам?"
        return None

    if intent == "losses":
        if "электросетевая компания" not in slots.get("object_type", "") and not slots.get("is_grid"):
            return "Вы электросетевая компания? Продукт предназначен именно для сетевых организаций."
        if not slots.get("has_hourly") and not slots.get("daily_only"):
            return "Какие данные учёта есть: почасовые интервалы или только суточные с разбивкой день/ночь?"
        if not slots.get("archive_years"):
            return "За какой период накоплен архив данных?"
        return None

    if intent == "reading":
        if not slots.get("resource"):
            return "Показания какого ресурса нужно распознавать: электроэнергия или вода?"
        if not slots.get("meters_count"):
            return "Сколько приборов или какой объём фотографий планируете обрабатывать?"
        if not slots.get("collection"):
            return "Как показания собираются сейчас: обходы, ручной ввод, звонки абонентов?"
        return None

    if intent == "orem":
        power = slots.get("power_kw")
        if not power:
            return "Какая у предприятия присоединённая мощность? Это ключевой параметр для выхода на оптовый рынок."
        if power < 750:
            return None
        if not slots.get("consumption_kwh"):
            return "Какой примерно объём месячного потребления?"
        if not slots.get("voltage") and not slots.get("price_category"):
            return "Какой у вас уровень напряжения и действующая ценовая категория?"
        return None

    if intent == "general":
        if not goals and not slots.get("object_type"):
            return "Какая задача стоит перед вами: снизить платёж, наладить учёт, найти потери или обеспечить надёжность?"
        return None

    return "Опишите, пожалуйста, задачу и тип объекта — подберу подходящее решение."


# Сборка локального ответа

PRODUCT_INTRO = {
    "solar": "По солнечной станции",
    "accounting": "По платформе учёта",
    "losses": "По поиску коммерческих потерь",
    "reading": "По распознаванию показаний",
    "orem": "По выходу на оптовый рынок",
    "general": "По вашему вопросу",
}

LEAD_CTA = "Заполните короткую форму — передам заявку инженеру, он подготовит бесплатный расчёт."


def _format_power(value: int | None) -> str:
    if not value:
        return ""
    if value >= 1000:
        return f"{value / 1000:g} МВт"
    return f"{value:g} кВт"


def _orem_intro(power: int | None) -> str:
    if power and power < 20_000:
        return (
            f"При мощности {_format_power(power)} участие в оптовом рынке возможно через "
            "энергосбытовую компанию-посредника: прямым участником рынка предприятие не становится, "
            "но экономический эффект достигается."
        )
    if power:
        return (
            f"При мощности {_format_power(power)} возможно прямое участие в оптовом рынке. "
            "Субъектом рынка остаётся само предприятие, МТС сопровождает учёт и процедуры."
        )
    return (
        "Ключевой параметр для выхода на оптовый рынок — присоединённая мощность. "
        "Ниже 750 кВт выход невозможен, от 750 кВт до 20 МВт — участие через посредника, "
        "от 20 МВт — прямое участие."
    )


def build_local_answer(
    question: str,
    state: dict,
    results: Sequence[tuple[Chunk, float]],
    *,
    force_lead: bool = False,
) -> str:
    """Ответ по курируемым правилам или найденным фрагментам базы знаний."""
    intent = state.get("intent", "unknown")
    slots = state.get("slots") or {}
    current = current_question(question)
    context = conversation_context(question)
    lowered = normalize(current)

    if is_greeting(current):
        return (
            "Здравствуйте! Я ИИ-консультант МТС EnergyTool. Помогу разобраться с учётом, "
            "солнечной генерацией, потерями или выходом на оптовый рынок. "
            "С какой задачей вы пришли?"
        )

    if is_thanks(current) and not force_lead:
        return "Рад помочь. Если нужно, могу передать задачу инженеру — он подготовит расчёт по вашему объекту."

    if is_out_of_scope(current):
        return (
            "Этот вопрос выходит за границы моей компетенции: я консультирую только по продуктам "
            "МТС EnergyTool и не даю юридических заключений и технических инструкций по приборам. "
            "Лучше передать вопрос специалисту — он ответит предметно. Оставить заявку?"
        )

    if wants_lead(current):
        return (
            "Хорошо, оформлю заявку. Заполните короткую форму: ФИО, телефон и компания. "
            "Инженер свяжется с вами, уточнит параметры объекта и подготовит бесплатный расчёт."
        )

    # Регуляторное ограничение важнее любых общих формулировок.
    if intent == "orem":
        power = slots.get("power_kw")
        if power and power < 750:
            return (
                "При присоединённой мощности ниже 750 кВт выход на оптовый рынок невозможен — "
                "это регуляторное ограничение. Но снизить затраты можно другими способами: "
                "разобрать ценовую категорию, наладить учёт или рассмотреть солнечную генерацию. "
                "Какой у вас профиль потребления и что хотелось бы изменить в счёте?"
            )

    # Сначала ищем правило под текущую реплику, затем — под весь контекст диалога.
    curated = curated_answer(current, intent, slots) or curated_answer(context, intent, slots)
    if curated:
        body, ask = curated
        parts = [body]
        if force_lead:
            parts.append(LEAD_CTA)
        elif ask:
            parts.append(ask)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    facts = pick_facts(current, intent, results, limit=2)
    if not facts:
        for chunk, _score in results[:2]:
            facts = sentences(chunk.text)[:2]
            if facts:
                break
    if not facts and results:
        facts = [results[0][0].excerpt(300)]

    if intent == "orem":
        intro = _orem_intro(slots.get("power_kw"))
        body = " ".join(facts[:1])
        answer = " ".join(part for part in (intro, body) if part)
    else:
        intro = PRODUCT_INTRO.get(intent, "По вашему вопросу")
        body = " ".join(facts)
        answer = f"{intro}: {body}" if body else intro

    # Аккуратные оговорки для чувствительных тем.
    caveats: list[str] = []
    if intent in ("solar", "orem", "accounting") and any(
        word in lowered for word in ("стоимост", "цена", "стоит", "бюджет", "окупа", "экономи")
    ):
        caveats.append("Точную стоимость и эффект называем после бесплатного расчёта по данным объекта.")
    if intent == "losses" and any(word in lowered for word in ("точно", "гарант", "хищен", "воров")):
        caveats.append(
            "Результат носит вероятностный характер: список приоритетных проверок, "
            "а не установленный факт хищения."
        )

    tail = next_question(intent, slots)
    if force_lead:
        tail = LEAD_CTA
    elif not tail and not state.get("lead_offered"):
        tail = "Могу передать задачу инженеру, чтобы подготовили расчёт по вашему объекту."

    if caveats:
        answer += " " + " ".join(caveats)
    if tail:
        answer += " " + tail
    return re.sub(r"\s+", " ", answer).strip()


# План хода и заявка


@dataclass
class TurnPlan:
    intent: str
    slots: dict
    question: str | None
    offer_lead: bool
    quick_replies: list[str] = field(default_factory=list)
    product: str = ""
    query: str = ""
    context: str = ""


def plan_turn(question: str, state: dict) -> TurnPlan:
    query = current_question(question)
    context = conversation_context(question)
    slots = extract_slots(context, state)
    previous_intent = state.get("intent", "unknown")
    detected = detect_intent(context, previous_intent)
    intent = detected if detected != "unknown" else previous_intent

    if intent == "unknown" or intent not in INTENT_LABELS:
        intent = "general"

    state["intent"] = intent
    state["product"] = INTENT_PRODUCTS.get(intent, "")
    state["slots"] = slots
    state["turn"] = int(state.get("turn", 0)) + 1
    topics = list(state.get("topics") or [])
    if intent not in topics:
        topics.append(intent)
    state["topics"] = topics

    explicit_lead = wants_lead(query)
    offer_lead = bool(explicit_lead or (state["turn"] >= 3 and not state.get("lead_offered")))
    if state.get("lead_done"):
        offer_lead = False

    if offer_lead:
        state["lead_offered"] = True
        state["stage"] = "lead_offered"

    question_to_ask = None if explicit_lead else next_question(intent, slots)

    return TurnPlan(
        intent=intent,
        slots=slots,
        question=question_to_ask,
        offer_lead=offer_lead,
        quick_replies=quick_replies(intent, slots, offer_lead),
        product=state["product"],
        query=query,
        context=context,
    )


def quick_replies(intent: str, slots: dict, offer_lead: bool) -> list[str]:
    if offer_lead:
        return ["Оставить заявку", "Продолжить консультацию"]
    if intent == "solar":
        if not slots.get("goals"):
            return ["Снизить платёж", "Нужен резерв при отключениях"]
        return ["Сетевая станция", "Гибридная с накопителями", "Как считается окупаемость?"]
    if intent == "accounting":
        return ["Учёт по цехам", "Учёт по арендаторам", "Подключить существующие счётчики"]
    if intent == "losses":
        return ["Есть почасовые данные", "Только суточные", "Как проходит выездная проверка?"]
    if intent == "reading":
        return ["Электроэнергия", "Вода", "У нас архив фотографий"]
    if intent == "orem":
        return ["Мощность до 750 кВт", "От 750 кВт до 20 МВт", "Больше 20 МВт"]
    return ["Снизить счёт за электричество", "Наладить учёт", "Найти потери", "Выйти на оптовый рынок"]


def make_summary(state: dict, lead: dict, messages: Sequence[dict]) -> str:
    """Человекочитаемое резюме диалога для менеджера."""
    slots = state.get("slots") or {}
    intent = state.get("intent", "unknown")
    goals = slots.get("goals") or []
    lines = [
        "ЗАЯВКА С САЙТА МТС ENERGYTOOL",
        "",
        f"Клиент: {lead.get('name') or slots.get('name') or 'не указано'}",
        f"Компания: {lead.get('company') or slots.get('company') or 'не указана'}",
        f"Телефон: {lead.get('phone') or slots.get('phone') or 'не указан'}",
        "",
        f"Задача: {INTENT_LABELS.get(intent, 'не определена')}",
        f"Рекомендованный продукт: {INTENT_PRODUCTS.get(intent, 'уточнить на звонке')}",
    ]

    if slots.get("object_type"):
        lines.append(f"Тип объекта: {slots['object_type']}")
    if slots.get("region"):
        lines.append(f"Регион: {slots['region']}")
    if slots.get("power_kw"):
        lines.append(f"Присоединённая мощность: {_format_power(slots['power_kw'])}")
    if slots.get("consumption_kwh"):
        lines.append(f"Месячное потребление: {slots['consumption_kwh']:,} кВт·ч".replace(",", " "))
    if slots.get("roof"):
        lines.append(f"Кровля/площадка: {slots['roof']}")
    if slots.get("meters_count"):
        lines.append(f"Приборов учёта: {slots['meters_count']}")
    if slots.get("price_category"):
        lines.append(f"Ценовая категория: {slots['price_category']}")
    if slots.get("voltage"):
        lines.append(f"Уровень напряжения: {slots['voltage']}")
    if goals:
        lines.append(f"Цели клиента: {', '.join(goals)}")
    if slots.get("has_hourly"):
        lines.append("Данные учёта: почасовые")
    if slots.get("daily_only"):
        lines.append("Данные учёта: только суточные")
    if slots.get("archive_years"):
        lines.append(f"Глубина архива: {slots['archive_years']} год(а)")

    user_messages = [m["content"] for m in messages if m.get("role") == "user"][-4:]
    if user_messages:
        lines.extend(["", "Что обсуждали:"])
        for message in user_messages:
            clean = re.sub(r"\s+", " ", message).strip()
            if len(clean) > 220:
                clean = clean[:217] + "…"
            lines.append(f"— {clean}")

    lines.extend(
        [
            "",
            "Следующий шаг: связаться с клиентом, уточнить недостающие параметры "
            "и подготовить бесплатный предварительный расчёт.",
        ]
    )
    return "\n".join(lines)
