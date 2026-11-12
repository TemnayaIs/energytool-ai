"""Ядро поиска по базе знаний МТС EnergyTool.

Чистый Python без внешних зависимостей: используется и веб-прототипом,
и конкурсным solution.py. Реализует:

* загрузку Markdown/TXT из папки или ZIP;
* нарезку по заголовкам и абзацам с перекрытием;
* BM25-поиск с лёгкой нормализацией русских слов;
* расширение запроса бытовыми синонимами;
* повышение веса профильных документов под определённую задачу.

Модуль намеренно не содержит сетевых вызовов и секретов.
"""

from __future__ import annotations

import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Нормализация и токенизация

WORD_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

STOPWORDS = {
    "а", "без", "более", "больше", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "ваш", "ваша", "ваше", "ваши", "вдруг", "ведь", "во",
    "вот", "впрочем", "все", "всего", "всех", "вы", "где", "да", "даже", "для",
    "до", "другая", "другие", "других", "его", "ее", "ей", "еще", "если", "есть",
    "ею", "ещё", "же", "за", "зато", "здесь", "и", "из", "или", "им", "иногда",
    "их", "к", "как", "какая", "какие", "каких", "какой", "когда", "кого", "ком",
    "кому", "которая", "которые", "который", "которых", "кроме", "кто", "куда",
    "ли", "либо", "мне", "может", "можно", "мой", "моя", "мы", "на", "над",
    "надо", "наконец", "нас", "не", "него", "нее", "ней", "нельзя", "нет", "ни",
    "нибудь", "никогда", "ним", "них", "ничего", "но", "ну", "о", "об", "однако",
    "он", "она", "они", "оно", "от", "отчего", "очень", "по", "под", "после",
    "потом", "потому", "почти", "при", "про", "раз", "разве", "с", "сам", "свою",
    "себе", "себя", "сейчас", "со", "совсем", "так", "также", "такой", "там",
    "тебя", "тем", "теперь", "то", "тогда", "того", "тоже", "той", "только",
    "том", "ты", "у", "уже", "хотя", "чего", "чей", "чем", "через", "что", "чтоб",
    "чтобы", "чуть", "эта", "эти", "этим", "этих", "это", "этого", "этой", "этом",
    "этот", "эту", "я",
}

# Бытовые формулировки -> термины базы знаний.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("солнечная", "солнечной", "солнечную", "сэс", "панели", "панель", "фотоэлектрическая", "гелио"),
    ("счетчик", "счётчик", "счетчика", "счётчика", "прибор", "прибора", "приборов", "учета", "учёта", "аскуэ"),
    ("хищение", "хищения", "воровство", "воруют", "незаконное", "подключение", "потери", "потерь", "небаланс"),
    ("фото", "фотография", "фотографии", "снимок", "камера", "телеграм", "telegram", "распознавание"),
    ("опт", "оптовый", "оптовом", "орэм", "рынок", "мощность", "мегаватт"),
    ("тариф", "тарифа", "тарифы", "ценовая", "категория", "категории", "цк"),
    ("напряжение", "напряжения", "вн", "сн", "нн", "киловольт", "кв"),
    ("аккумулятор", "накопитель", "накопители", "гибридная", "гибрид", "автономность", "резерв"),
    ("окупаемость", "окупается", "окупится", "экономия", "эффект", "выгода"),
    ("стоимость", "цена", "цены", "стоит", "бюджет", "смета"),
    ("рассрочка", "оплата", "платеж", "платёж", "лизинг"),
    ("арендатор", "арендаторы", "арендаторов", "управляющая", "тсж", "снт", "товарищество"),
    ("потери", "потерь", "хищение", "небаланс", "выезд", "проверка"),
    ("реактивная", "реактивной", "реактив", "компенсация", "компенсировать", "компенсирующие"),
    ("мощность", "квт", "киловатт", "присоединенная", "присоединённая"),
    ("фасад", "фасаде", "фасада", "вертикальная", "вертикальные", "вертикальных", "поверхность", "поверхности"),
    ("симка", "симки", "сим-карта", "сим-карты", "сим-картами", "сим"),
    ("двухтарифный", "двухтарифные", "многотарифный", "многотарифных", "тариф", "тарифам", "тарифная", "тарифные"),
    ("год выпуска", "год производства", "год", "выпуска", "производства"),
    ("магазин", "магазины", "магазинов", "сеть", "сетевой", "точка", "точки", "объект", "объекты"),
    ("цех", "цеха", "цехам", "подразделение", "подразделения", "подразделениям"),
    ("кабинет", "кабинете", "личный", "платформа", "платформе"),
    ("невыгоден", "невыгодно", "невыгодный", "окупаемость", "окупается", "экономия", "эффект"),
    ("расширить", "расширение", "расширяться", "нарастить", "наращивается", "поэтапно", "поэтапный"),
    ("категория", "категории", "категорию", "цк", "ценовая"),
)

SUFFIXES = sorted(
    {
        "иями", "ями", "ами", "иях", "ях", "ах", "его", "ого", "ему", "ому",
        "ение", "ание", "ость", "есть", "еть", "ить", "ать", "ять", "ыть",
        "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
        "ей", "ем", "ом", "ам", "ям", "ов", "ев", "ия", "ию", "ии", "ье",
        "ы", "и", "а", "о", "у", "е", "ю", "я", "ь",
    },
    key=len,
    reverse=True,
)


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру и убирает пунктуацию."""
    return " ".join(WORD_RE.findall(text.lower().replace("ё", "е")))


def stem(word: str) -> str:
    """Очень лёгкий стеммер: срезает один частый русский суффикс."""
    if len(word) <= 4 or not word.isalpha():
        return word
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens: list[str] = []
    for raw in normalize(text).split():
        if not keep_stopwords and raw in STOPWORDS:
            continue
        if len(raw) < 2:
            continue
        tokens.append(stem(raw))
    return tokens


def expand_tokens(tokens: Sequence[str]) -> list[str]:
    """Добавляет синонимы из бытовых формулировок."""
    expanded = list(tokens)
    token_set = set(tokens)
    for group in SYNONYM_GROUPS:
        group_stems = {stem(normalize(word)) for word in group}
        if token_set & group_stems:
            for candidate in group_stems:
                if candidate not in token_set:
                    expanded.append(candidate)
                    token_set.add(candidate)
    return expanded


# Документы и фрагменты


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    heading: str
    text: str
    tokens: list[str] = field(default_factory=list)

    @property
    def location(self) -> str:
        return f"{self.title} → {self.heading}" if self.heading and self.heading != self.title else self.title

    def excerpt(self, limit: int = 420) -> str:
        text = re.sub(r"\|?[\s:—-]{3,}\|?", " ", self.text)
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(line.strip(" |") for line in text.splitlines() if line.strip(" |-"))
        text = " ".join(text.split())
        if len(text) > limit:
            cut = text[:limit].rsplit(" ", 1)[0]
            return cut + "…"
        return text


def _document_title(name: str, text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line and not line.startswith(("---", "|")):
            return line[:120]
    return Path(name).stem


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Возвращает пары (путь заголовков, тело раздела)."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            path = " → ".join(title for _, title in stack)
            sections.append((path, content))
        body.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            stack[:] = [item for item in stack if item[0] < level]
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def _split_blocks(content: str, max_chars: int = 1100, overlap: int = 240) -> list[str]:
    """Режет раздел по абзацам, сохраняя таблицы и списки целиком."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    if not blocks:
        return []

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars * 1.6:
            # Слишком длинный блок (например, большая таблица) режем по строкам.
            lines = block.splitlines()
            piece = ""
            for line in lines:
                if len(piece) + len(line) + 1 > max_chars and piece:
                    chunks.append(piece.strip())
                    piece = ""
                piece += line + "\n"
            if piece.strip():
                chunks.append(piece.strip())
            continue

        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = (tail + "\n\n" + block).strip() if tail else block
        else:
            current = (current + "\n\n" + block).strip() if current else block

    if current.strip():
        chunks.append(current.strip())

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars * 1.8:
            result.append(chunk)
        else:
            for start in range(0, len(chunk), max_chars):
                part = chunk[start : start + max_chars].strip()
                if part:
                    result.append(part)
    return result


def chunks_from_text(name: str, text: str) -> list[Chunk]:
    title = _document_title(name, text)
    chunks: list[Chunk] = []
    for section_index, (heading, body) in enumerate(_split_sections(text)):
        for part_index, part in enumerate(_split_blocks(body)):
            chunks.append(
                Chunk(
                    id=f"{name}::{section_index}::{part_index}",
                    source=name,
                    title=title,
                    heading=heading or title,
                    text=part,
                )
            )
    if not chunks:
        chunks.append(Chunk(id=f"{name}::0::0", source=name, title=title, heading=title, text=text.strip()))
    for chunk in chunks:
        chunk.tokens = tokenize(f"{chunk.title} {chunk.heading} {chunk.text}")
    return chunks


def iter_documents(path: str | Path) -> Iterable[tuple[str, str]]:
    """Читает Markdown/TXT из папки или ZIP-архива базы знаний."""
    path = Path(path)
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix.lower() in (".md", ".txt"):
                yield file.relative_to(path).as_posix(), file.read_text(encoding="utf-8-sig", errors="replace")
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith("__MACOSX/") or Path(name).suffix.lower() not in (".md", ".txt"):
                    continue
                yield name, archive.read(name).decode("utf-8-sig", errors="replace")
        return
    raise FileNotFoundError(f"База знаний не найдена: {path}")


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, text in iter_documents(path):
        if text.strip():
            chunks.extend(chunks_from_text(name, text))
    if not chunks:
        raise ValueError("В базе знаний нет непустых .md/.txt файлов")
    return chunks


# BM25


class BM25Index:
    """BM25 с бонусами за заголовок, точную фразу и профильный документ."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [chunk.tokens or tokenize(f"{chunk.title} {chunk.heading} {chunk.text}") for chunk in self.chunks]
        self.doc_sets = [set(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in self.doc_tokens]
        document_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_freq.update(set(tokens))
        total = max(1, len(self.doc_tokens))
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def _score(self, query_terms: Sequence[str], index: int) -> float:
        freqs = self.term_freqs[index]
        length = self.doc_len[index] or 1
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = tf + self.k1 * (1 - self.b + self.b * length / max(1.0, self.avg_len))
            score += idf * tf * (self.k1 + 1) / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int = 6,
        *,
        preferred_sources: Sequence[str] = (),
        demoted_sources: Sequence[str] = (),
        min_score: float = 0.0,
        phrase_query: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_terms = expand_tokens(tokenize(query))
        if not query_terms:
            return []

        query_norm = " ".join(normalize(phrase_query if phrase_query is not None else query).split())
        phrase = query_norm if len(query_norm) >= 8 else ""
        query_set = set(query_terms)
        preferred = tuple(preferred_sources)
        demoted = tuple(demoted_sources)

        scored: list[tuple[Chunk, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)

            # Бонус за совпадение с заголовком раздела.
            heading_terms = set(tokenize(f"{chunk.title} {chunk.heading}"))
            score += 0.9 * len(query_set & heading_terms)

            # Покрытие редких терминов запроса: важнее, чем частое слово.
            coverage = sum(self.idf.get(term, 0.0) for term in query_set & self.doc_sets[index])
            score += 0.22 * coverage

            # Бонус за точную фразу в тексте.
            if phrase and phrase in normalize(chunk.text):
                score += 2.2

            # Профильный документ под текущую задачу.
            if preferred and any(source in chunk.source for source in preferred):
                score += 1.8
            # Чужой продуктовый документ не должен перебивать профильный
            # только за счёт общих слов вроде «сколько это стоит».
            elif demoted and any(source in chunk.source for source in demoted):
                score -= 1.2

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        # Убираем почти одинаковые фрагменты, чтобы контекст не дублировался.
        selected: list[tuple[Chunk, float]] = []
        seen: set[tuple[str, str]] = set()
        for chunk, score in scored:
            key = (chunk.title, chunk.text[:160])
            if key in seen:
                continue
            seen.add(key)
            selected.append((chunk, score))
            if len(selected) >= top_k:
                break

        if min_score > 0:
            selected = [(chunk, score) for chunk, score in selected if score >= min_score]
        return selected


# Вспомогательные функции для контекста и источников

INTENT_SOURCES: dict[str, tuple[str, ...]] = {
    "solar": ("23-solar", "20-portfolio", "03-calculation"),
    "accounting": ("22-askue", "20-portfolio", "26-general"),
    "losses": ("24-losses", "20-portfolio"),
    "reading": ("25-meter-reading", "20-portfolio"),
    "orem": ("21-orem", "12-power-thresholds", "11-price-categories", "20-portfolio"),
    "general": ("26-general", "20-portfolio", "02-product-scope", "03-calculation"),
}

# Тематические подсказки для BM25: интент уже известен, поэтому полезно
# усилить лексику профильного направления, даже если клиент говорит бытовыми словами.
INTENT_QUERY_HINTS: dict[str, str] = {
    "solar": "солнечная станция панели выработка кровля",
    "accounting": "платформа учёта приборы показания кабинет",
    "losses": "потери хищения выездная проверка",
    "reading": "распознавание показаний фотография",
    "orem": "оптовый рынок присоединённая мощность",
    "general": "",
}


def search(
    index: BM25Index,
    query: str,
    *,
    intent: str = "general",
    top_k: int = 6,
    min_score: float = 0.0,
) -> list[tuple[Chunk, float]]:
    hint = INTENT_QUERY_HINTS.get(intent, "")
    effective_query = f"{query} {hint}".strip() if hint else query
    return index.search(
        effective_query,
        top_k=top_k,
        preferred_sources=INTENT_SOURCES.get(intent, INTENT_SOURCES["general"]),
        demoted_sources=("20-products/",) if intent != "general" else (),
        min_score=min_score,
        phrase_query=query,
    )


def build_context(results: Sequence[tuple[Chunk, float]], limit_chars: int = 6500) -> str:
    parts: list[str] = []
    used = 0
    for chunk, _score in results:
        block = f"[{chunk.source}] {chunk.heading}\n{chunk.text.strip()}"
        if used + len(block) > limit_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def sources_payload(results: Sequence[tuple[Chunk, float]], limit: int = 5) -> list[dict]:
    return [
        {
            "file": chunk.source,
            "title": chunk.title,
            "heading": chunk.heading,
            "score": round(float(score), 3),
            "excerpt": chunk.excerpt(360),
        }
        for chunk, score in results[:limit]
    ]
