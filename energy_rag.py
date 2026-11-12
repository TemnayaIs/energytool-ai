"""Ядро ИИ-консультанта МТС EnergyTool: поиск по базе знаний и генерация ответа.

Один модуль используют и конкурсный ``solution.py``, и веб-прототип ``app/``,
поэтому качество ответов на лидерборде и в чате одинаковое.

Пайплайн одного ответа:

1. Разбор входа: последняя реплика клиента + история диалога.
2. Гибридный поиск: BM25 (стемминг, синонимы) и плотные эмбеддинги BAAI/bge-m3,
   отдельно по последней реплике и по всему контексту диалога; слияние RRF.
3. Реранжирование кандидатов Qwen3 Reranker (если доступен).
4. Сборка контекста: найденные разделы + «правила продукта» (разделы
   «Факты, которые можно утверждать» / «Чего утверждать нельзя»)
   для продуктов, попавших в выдачу.
5. Генерация ответа консультанта (Qwen3, без «размышлений»), очистка разметки.
6. Любой сбой сети -> безопасный экстрактивный ответ, а не падение.

Зависимости: только ``httpx``. Секреты читаются из окружения и нигде не логируются.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import httpx

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


def env(*names: str, default: str = "") -> str:
    """Первое непустое значение из списка переменных окружения."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def env_number(name: str, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name, default=str(default)))))
    except (TypeError, ValueError):
        return default


def env_flag(name: str, default: bool) -> bool:
    value = env(name, default="1" if default else "0").lower()
    return value not in ("0", "false", "no", "off", "")


@dataclass
class Settings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    emb_model: str = ""
    reranker_model: str = ""
    temperature: float = 0.1
    max_tokens: int = 700
    top_k: int = 8
    rerank_candidates: int = 24
    timeout: float = 60.0
    use_embeddings: bool = True
    use_reranker: bool = True
    emb_base_url: str = ""  # отдельный сервер эмбеддингов (локальные тесты); пусто — общий прокси
    emb_api_key: str = ""
    enable_thinking: bool = False  # режим размышлений Qwen3 (эксперимент)
    self_check: bool = False  # второй вызов: сверка ответа с контекстом (эксперимент)

    @classmethod
    def from_env(cls) -> "Settings":
        temperature = env_number("TEMPERATURE", 0.1, 0.0, 2.0, float)
        max_tokens = env_number("MAX_TOKENS", 700, 64, 8192)
        return cls(
            base_url=env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/"),
            api_key=env("LLM_API_KEY", "OPENAI_API_KEY"),
            model=env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default="qwen/qwen3.8-27b"),
            emb_model=env("EMB_MODEL", default="BAAI/bge-m3"),
            reranker_model=env("RERANKER_MODEL", default="Qwen/Qwen3-Reranker-8B"),
            # Консультанту нужна воспроизводимость: высокую температуру из
            # конфигурации ограничиваем сверху.
            temperature=min(temperature, 0.3),
            # Ответ ~150 слов по-русски — это 300–450 токенов; оставляем запас,
            # но не даём модели «расписаться».
            max_tokens=max(450, min(max_tokens, 1200)),
            top_k=max(6, env_number("TOP_K", 8, 3, 16)),  # слишком малый TOP_K режет нужные разделы
            rerank_candidates=max(env_number("RERANKER_TOP_N", 24, 4, 40), 16),
            timeout=float(env_number("LLM_TIMEOUT", 60, 10, 180)),
            use_embeddings=env_flag("USE_EMBEDDINGS", True),
            use_reranker=env_flag("USE_RERANKER", True),
            emb_base_url=env("EMB_BASE_URL").rstrip("/"),
            emb_api_key=env("EMB_API_KEY"),
            enable_thinking=env_flag("ENABLE_THINKING", False),
            self_check=env_flag("SELF_CHECK", False),
        )

    @property
    def llm_ready(self) -> bool:
        return bool(self.base_url and self.api_key)


# ---------------------------------------------------------------------------
# Нормализация текста и BM25
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

STOPWORDS = set(
    """а без бы был была были было быть в вам вас ваш ваша ваше ваши во вот все всё всего
    всех вы где да даже для до его ее её ей если есть еще ещё же за здесь и из или им их
    к как какая какие какой когда кто ли либо мне может можно мой мы на над надо нам нас
    не него нее неё ней нет ни но ну о об он она они оно от по под при про с со так также
    там те тем то того тоже той только том ты у уже чем что чтобы эта эти это этого этой
    этом этот эту я вообще просто нужно нужен нужна хотим хочу у нас наш наша наше наши""".split()
)

# Бытовые формулировки клиента -> лексика базы знаний (до стемминга).
SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("солнечная", "солнечные", "сэс", "панели", "фотоэлектрическая", "генерация", "станция"),
    ("счетчик", "счетчики", "прибор", "приборы", "учет", "аскуэ"),
    ("хищение", "хищения", "воровство", "воруют", "потери", "небаланс", "безучетное"),
    ("фото", "фотография", "фотографии", "снимок", "распознавание", "показания"),
    ("опт", "оптовый", "орэм", "рынок", "энергосбытовая"),
    ("тариф", "тарифы", "ценовая", "категория", "цк"),
    ("аккумулятор", "аккумуляторы", "накопитель", "накопители", "гибридная", "автономность", "резерв"),
    ("окупаемость", "окупается", "окупится", "экономия", "эффект", "выгода", "невыгодно"),
    ("стоимость", "цена", "стоит", "бюджет", "смета", "сколько"),
    ("рассрочка", "оплата", "платеж", "лизинг", "капитальные"),
    ("арендатор", "арендаторы", "бизнесцентр", "управляющая", "тсж", "снт"),
    ("реактивная", "реактивной", "компенсация", "компенсирующие"),
    ("мощность", "квт", "киловатт", "присоединенная", "мвт"),
    ("фасад", "вертикальные", "вертикальных", "стена"),
    ("симка", "симки", "сим", "sim", "оператор", "операторов", "связь"),
    ("двухтарифный", "двухтарифные", "многотарифный", "многотарифных", "тарифные", "зоны"),
    ("магазин", "магазины", "ретейл", "сеть", "точки", "филиалы", "склады"),
    ("цех", "цеха", "цехам", "подразделения", "оборудование"),
    ("кабинет", "личный", "платформа", "интерфейс", "облачная"),
    ("свет", "электричество", "электроэнергия", "энергия"),
    ("счет", "счета", "квитанция", "платеж", "сумма"),
    ("кровля", "крыша", "скат", "скаты", "шифер", "площадка"),
    ("проверите", "проверить", "аудит", "аудита", "обследование", "обследования"),
    ("расширить", "расширение", "расширяться", "нарастить", "наращивается", "наращивать", "поэтапно", "поэтапный", "очередями"),
    ("начать", "начинать", "начинается", "старт", "пилот", "пилотное"),
    ("данные", "архив", "выгрузка", "глубина"),
    ("вечер", "вечером", "вечерняя", "ночь", "профиль", "часы"),
)

_SUFFIXES = sorted(
    """иями ями ами иях ях ах его ого ему ому ыми ими ение ание ость ться тся еть ить ать ять
    ую юю ая яя ое ее ые ие ый ий ой ей ем ом ам ям ов ев ия ию ии ье ых их ы и а о у е ю я ь""".split(),
    key=len,
    reverse=True,
)


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower().replace("ё", "е")))


def stem(word: str) -> str:
    if len(word) <= 4 or not word.isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in normalize(text).split() if len(t) > 1 and t not in STOPWORDS]


_SYN_STEMS = [frozenset(stem(normalize(w)) for w in group) for group in SYNONYMS]


def expand(tokens: Sequence[str]) -> list[str]:
    out = list(tokens)
    have = set(tokens)
    for group in _SYN_STEMS:
        if have & group:
            for t in group - have:
                out.append(t)
                have.add(t)
    return out


# ---------------------------------------------------------------------------
# База знаний
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    source: str  # путь файла внутри базы
    title: str  # заголовок документа
    heading: str  # путь заголовков раздела
    text: str
    tokens: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.title} → {self.heading}" if self.heading and self.heading != self.title else self.title

    def for_embedding(self) -> str:
        return f"{self.label}\n{self.text}"

    def excerpt(self, limit: int = 360) -> str:
        text = re.sub(r"\s+", " ", re.sub(r"[|]|-{3,}", " ", self.text)).strip()
        return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _iter_documents(path: Path) -> Iterable[tuple[str, str]]:
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


def _clean_md(text: str) -> str:
    # Pandoc-таблицы и разделители не несут смысла для модели.
    text = re.sub(r"(?m)^\s*-{3,}[\s-]*$", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = _clean_md("\n".join(body))
        if content:
            # Заголовок документа (H1) хранится отдельно, в путь раздела не входит.
            path = " → ".join(t for lvl, t in stack if lvl > 1)
            sections.append((path, content))
        body.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = [item for item in stack if item[0] < level]
            stack.append((level, m.group(2).strip()))
        else:
            body.append(line)
    flush()
    return sections


def _split(content: str, max_chars: int = 1600, overlap: int = 200) -> list[str]:
    if len(content) <= max_chars:
        return [content]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    parts: list[str] = []
    current = ""
    for block in blocks:
        while len(block) > max_chars:  # очень длинный абзац/таблица
            cut = block.rfind("\n", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                parts.append(current)
                current = ""
            parts.append(block[:cut].strip())
            block = block[cut:].strip()
        if current and len(current) + len(block) + 2 > max_chars:
            parts.append(current)
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else ""
            current = (tail + "\n\n" + block).strip() if tail else block
        else:
            current = f"{current}\n\n{block}".strip() if current else block
    if current:
        parts.append(current)
    return parts


GUARD_HEADINGS = ("факты, которые можно утверждать", "чего утверждать нельзя", "что мы не делаем")


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    guards: dict[str, str]  # source -> текст правил продукта
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        path = Path(path)
        chunks: list[Chunk] = []
        guards: dict[str, list[str]] = defaultdict(list)
        digest = hashlib.sha1()
        for name, raw in _iter_documents(path):
            if not raw.strip():
                continue
            digest.update(name.encode())
            digest.update(raw.encode())
            title = next(
                (ln.lstrip("#").strip() for ln in raw.splitlines() if ln.startswith("# ")),
                Path(name).stem,
            )
            for s_idx, (heading, body) in enumerate(_sections(raw)):
                if heading.lower().split(" → ")[0] in GUARD_HEADINGS:
                    guards[name].append(f"{heading}:\n{body}")
                for p_idx, part in enumerate(_split(body)):
                    chunk = Chunk(f"{name}#{s_idx}.{p_idx}", name, title, heading or title, part)
                    chunk.tokens = tokenize(f"{title} {heading} {heading} {part}")
                    chunks.append(chunk)
        if not chunks:
            raise ValueError("В базе знаний нет непустых .md/.txt файлов")
        return cls(chunks, {k: "\n\n".join(v) for k, v in guards.items()}, digest.hexdigest()[:16])


class BM25:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.4, b: float = 0.7):
        self.k1, self.b = k1, b
        self.tf = [Counter(c.tokens) for c in chunks]
        self.len = [len(c.tokens) or 1 for c in chunks]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter[str] = Counter()
        for c in chunks:
            df.update(set(c.tokens))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        weights = Counter(terms)
        for t in expand(terms):
            weights.setdefault(t, 0.5)  # синонимы весят меньше исходных слов
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg)
            for term, w in weights.items():
                f = tf.get(term)
                if f:
                    s += w * self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Клиент OpenAI-совместимого прокси
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    pass


class ProxyClient:
    """Минимальный клиент с ограниченными повторами (429/5xx/таймауты)."""

    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, settings: Settings, retries: int = 3):
        self.s = settings
        self.retries = retries
        self.http = httpx.Client(
            timeout=httpx.Timeout(settings.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        self.no_think_kwargs = True  # передавать chat_template_kwargs, пока прокси не отверг
        self._lock = threading.Lock()

    def close(self) -> None:
        self.http.close()

    def post(
        self,
        endpoint: str,
        payload: dict,
        *,
        timeout: float | None = None,
        retries: int | None = None,
        base_url: str = "",
        api_key: str = "",
    ) -> dict:
        if not self.s.llm_ready:
            raise LLMError("LLM_BASE_URL/LLM_API_KEY не заданы")
        url = (base_url or self.s.base_url) + endpoint
        headers = {"Authorization": f"Bearer {api_key or self.s.api_key}"}
        attempts = self.retries if retries is None else retries
        last = "unknown"
        for attempt in range(attempts):
            try:
                r = self.http.post(url, json=payload, headers=headers, timeout=timeout or self.s.timeout)
                if r.status_code in self.RETRYABLE:
                    last = f"HTTP {r.status_code}"
                    retry_after = r.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 1.5 * (attempt + 1)
                    time.sleep(min(delay, 8.0))
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                return r.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = type(e).__name__
                time.sleep(1.0 * (attempt + 1))
            except ValueError as e:  # невалидный JSON
                last = f"bad json: {e}"
        raise LLMError(f"{endpoint}: {last}")

    # --- эмбеддинги и реранкер ---------------------------------------------

    def embed(self, texts: Sequence[str], batch: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            part = list(texts[start : start + batch])
            data = self.post(
                "/embeddings",
                {"model": self.s.emb_model, "input": part},
                timeout=60,
                base_url=self.s.emb_base_url,
                api_key=self.s.emb_api_key,
            )
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            if len(items) != len(part):
                raise LLMError("embeddings: размер ответа не совпадает")
            vectors.extend(_unit(item["embedding"]) for item in items)
        return vectors

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        payload = {
            "model": self.s.reranker_model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": False,
        }
        data = self.post("/rerank", payload, timeout=40, retries=2)
        entries = data.get("results") or data.get("data") or []
        out = []
        for e in entries:
            idx = int(e["index"])
            score = float(e.get("relevance_score", e.get("score", 0.0)))
            if 0 <= idx < len(documents):
                out.append((idx, score))
        if not out:
            raise LLMError("rerank: пустой ответ")
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # --- генерация -----------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> str:
        payload = {
            "model": self.s.model,
            "messages": messages,
            "temperature": self.s.temperature if temperature is None else temperature,
            "max_tokens": (max_tokens or self.s.max_tokens) + (3000 if self.s.enable_thinking else 0),
        }
        sent_kwargs = self.no_think_kwargs
        if sent_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": self.s.enable_thinking}
        try:
            data = self.post("/chat/completions", payload, timeout=timeout, retries=retries)
        except LLMError as e:
            # Прокси может не знать chat_template_kwargs — повторяем без него
            # и включаем «мягкий» выключатель размышлений Qwen3.
            status = str(e)[:8]
            if sent_kwargs and status.startswith("HTTP 4") and status not in ("HTTP 401", "HTTP 403"):
                with self._lock:
                    self.no_think_kwargs = False
                payload.pop("chat_template_kwargs", None)
                msgs = [dict(m) for m in messages]
                if not self.s.enable_thinking:
                    msgs[-1]["content"] = msgs[-1]["content"] + "\n/no_think"
                payload["messages"] = msgs
                data = self.post("/chat/completions", payload, timeout=timeout, retries=retries)
            else:
                raise
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("chat: неожиданный формат ответа") from e
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        content = strip_reasoning(content)
        if not content.strip():
            raise LLMError("chat: пустой ответ (возможно, токены ушли на размышления)")
        return content


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def strip_reasoning(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    if "</think>" in text:  # открывающий тег съеден шаблоном
        text = text.split("</think>", 1)[1]
    if "<think>" in text:  # размышление не закончилось — ответа нет
        text = text.split("<think>", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Разбор диалога
# ---------------------------------------------------------------------------

TURN_RE = re.compile(
    r"^\s*(пользователь|клиент|user|ассистент|консультант|бот|assistant)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Dialog:
    turns: list[tuple[str, str]]  # (role: user|assistant, text)

    @property
    def last_user(self) -> str:
        for role, text in reversed(self.turns):
            if role == "user":
                return text
        return self.turns[-1][1] if self.turns else ""

    @property
    def user_turns(self) -> list[str]:
        return [t for r, t in self.turns if r == "user"]

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1

    def history_text(self) -> str:
        names = {"user": "Клиент", "assistant": "Консультант"}
        return "\n".join(f"{names[r]}: {t}" for r, t in self.turns[:-1])


def parse_dialog(text: str) -> Dialog:
    """Разбирает вопрос с встроенной историей «Пользователь: … Ассистент: …»."""
    text = text.replace("\r\n", "\n").strip()
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return Dialog([("user", " ".join(text.split()))])
    turns: list[tuple[str, str]] = []
    preamble = " ".join(text[: matches[0].start()].split())
    if preamble:
        turns.append(("user", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = " ".join(text[m.end() : end].split())
        if not body:
            continue
        role = "user" if m.group(1).lower() in ("пользователь", "клиент", "user") else "assistant"
        turns.append((role, body))
    if not turns:
        return Dialog([("user", text)])
    if turns[-1][0] != "user":  # последняя реплика должна быть клиента
        turns.append(("user", turns.pop()[1]))
    return Dialog(turns)


def dialog_from_messages(history: Sequence[dict], message: str) -> Dialog:
    turns = []
    for m in history:
        role = "user" if m.get("role") == "user" else "assistant"
        content = " ".join(str(m.get("content", "")).split())
        if content:
            turns.append((role, content))
    inner = parse_dialog(message)
    return Dialog(turns + inner.turns)


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str = ""


class Retriever:
    def __init__(self, kb: KnowledgeBase, client: ProxyClient | None, settings: Settings, cache_dir: str | Path | None = None):
        self.kb = kb
        self.client = client
        self.s = settings
        self.bm25 = BM25(kb.chunks)
        self.vectors: list[list[float]] | None = None
        self.dense_error = ""
        self.rerank_ok = settings.use_reranker
        if client and settings.llm_ready and settings.use_embeddings:
            self._init_vectors(cache_dir)

    def _init_vectors(self, cache_dir: str | Path | None) -> None:
        cache = None
        if cache_dir:
            cache = Path(cache_dir) / f"kb_emb_{self.kb.fingerprint}_{re.sub(r'[^a-z0-9]+', '-', self.s.emb_model.lower())}.json"
            if cache.is_file():
                try:
                    data = json.loads(cache.read_text())
                    if len(data) == len(self.kb.chunks):
                        self.vectors = data
                        return
                except (OSError, ValueError):
                    pass
        try:
            self.vectors = self.client.embed([c.for_embedding() for c in self.kb.chunks])
            if cache:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(self.vectors))
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001 — эмбеддинги необязательны
            self.vectors = None
            self.dense_error = f"{type(e).__name__}: {e}"[:200]

    @property
    def mode(self) -> str:
        parts = ["bm25"]
        if self.vectors is not None:
            parts.append("bge-m3")
        if self.rerank_ok and self.client and self.s.llm_ready:
            parts.append("rerank")
        return "+".join(parts)

    def _rank_bm25(self, query: str, n: int) -> list[int]:
        scores = self.bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]

    def _rank_dense(self, qvec: list[float], n: int) -> list[int]:
        sims = [_dot(qvec, v) for v in self.vectors]  # type: ignore[union-attr]
        return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:n]

    def search(self, dialog: Dialog, *, deadline: float | None = None) -> list[Hit]:
        last = dialog.last_user
        users = dialog.user_turns
        # Запрос по контексту: последняя реплика + предыдущие реплики клиента
        # + последний вопрос консультанта (часто он задаёт тему ответа).
        context_parts = [last] + users[-3:-1][::-1]
        if dialog.is_multi_turn:
            prev_bot = next((t for r, t in reversed(dialog.turns[:-1]) if r == "assistant"), "")
            if prev_bot:
                context_parts.append(prev_bot)
        context_query = " ".join(context_parts)

        rankings: list[tuple[list[int], float]] = [(self._rank_bm25(last, 40), 1.0)]
        if dialog.is_multi_turn:
            rankings.append((self._rank_bm25(context_query, 40), 0.7))

        if self.vectors is not None and self.client and not _expired(deadline, 20):
            try:
                queries = [last] + ([context_query] if dialog.is_multi_turn else [])
                qvecs = self.client.embed(queries)
                rankings.append((self._rank_dense(qvecs[0], 40), 1.3))
                if len(qvecs) > 1:
                    rankings.append((self._rank_dense(qvecs[1], 40), 1.0))
            except Exception:  # noqa: BLE001
                pass

        fused: dict[int, float] = defaultdict(float)
        for ranking, weight in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] += weight / (60 + rank)
        order = sorted(fused, key=fused.get, reverse=True)
        candidates = order[: self.s.rerank_candidates]
        hits = [Hit(self.kb.chunks[i], fused[i], "rrf") for i in candidates]

        if self.rerank_ok and self.client and self.s.llm_ready and len(hits) > 2 and not _expired(deadline, 25):
            rq = last if not dialog.is_multi_turn else f"{last}\nКонтекст диалога: {context_query}"
            try:
                ranked = self.client.rerank(rq, [h.chunk.for_embedding()[:2000] for h in hits], top_n=len(hits))
                reranked = [Hit(hits[i].chunk, s, "rerank") for i, s in ranked]
                # Сохраняем разнообразие: лучший результат RRF всегда остаётся в выдаче.
                top = reranked[: self.s.top_k]
                if hits[0].chunk.id not in {h.chunk.id for h in top}:
                    top = top[:-1] + [hits[0]]
                return top
            except Exception:  # noqa: BLE001 — реранкер необязателен
                pass
        return hits[: self.s.top_k]


def _expired(deadline: float | None, reserve: float = 0.0) -> bool:
    return deadline is not None and time.monotonic() > deadline - reserve


# ---------------------------------------------------------------------------
# Промпт и генерация
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — ИИ-консультант МТС EnergyTool на сайте компании. К тебе приходят представители бизнеса, энергетики предприятий, председатели СНТ, сетевые организации и менеджеры МТС. Они пишут своими словами и часто не знают терминов.

Как ты отвечаешь:
1. Первое предложение — прямой ответ именно на заданный вопрос («Да, …», «Нет, …», «Да, но …», «Может иметь смысл, но …», «Начать стоит с …»). Не пересказывай вопрос и не здоровайся.
2. Затем 1–2 предложения по существу: только те факты из базы знаний, которые нужны для ответа на этот вопрос, и главная оговорка (от чего зависит результат или что нужно проверить). Не перечисляй всё, что знаешь о продукте: лишние цифры и характеристики, о которых не спрашивали, ухудшают ответ.
3. В конце — один уточняющий вопрос о главном параметре для следующего шага консультации (регион, сколько приборов и как собираются показания, профиль потребления днём и вечером, ценовая категория, возраст кровли и т. п.). Не спрашивай технические мелочи и то, что клиент уже сообщил. Если вопрос справочный и полностью закрыт ответом, уточняющий вопрос не нужен.
4. Объём — 2–4 предложения, обычно 35–70 слов. Обычный текст одним абзацем: без списков, заголовков, Markdown и ссылок на разделы базы.
5. Говори от лица МТС EnergyTool («мы», «наша платформа»).

Рассуждай как консультант, а не как поисковик:
- Свяжи факты базы с ситуацией клиента из диалога. Например, сетевая солнечная станция вырабатывает энергию днём и без накопителей не переносит её на вечер — значит, при вечернем потреблении эффект ниже и стоит сравнить с гибридной станцией.
- Не строй догадок о причинах ситуации клиента («скорее всего, у вас …»). Вместо этого скажи, от чего это может зависеть и что сравнить или проверить.
- Если в диалоге клиент уже ответил на вопрос консультанта, используй это и двигайся дальше.

Правила достоверности:
- Факты о продуктах, ценах, сроках, тарифах, порогах, законах и условиях бери ТОЛЬКО из КОНТЕКСТА. Не добавляй знаний «от себя», не округляй и не меняй цифры.
- Если в контексте нет ответа или вопрос не про продукты EnergyTool и связанные задачи клиента, прямо скажи, что в наших материалах этого нет / это вне компетенции консультанта, и предложи связаться со специалистом. Не выдумывай.
- Соблюдай раздел «ПРАВИЛА ПРОДУКТА»: не утверждай то, что там запрещено (итоговая стоимость, гарантированная экономия, точный срок окупаемости до расчёта, имена клиентов и производителей, внутренние алгоритмы, автономность сетевой станции и т. п.).
- Индикативные цифры называй только с оговоркой, что точный результат — после расчёта по объекту.
- Контакты (ФИО, телефон, компания) предлагай оставить, только если клиент сам хочет расчёт, звонок или специалиста. Если клиент сказал, что не готов оставлять контакты, — не проси их, просто помоги по существу.
- Технические инструкции по настройке приборов и юридические заключения — вне компетенции: предложи специалиста.

Примеры хороших ответов:

Клиент: У нас крыша на два ската — восток и запад. Панели можно поставить на оба?
Консультант: Да, при ориентации скатов на восток и запад можно использовать оба. Выработку мы рассчитываем с учётом региона, наклона и характеристик кровли. Подскажите, в каком регионе находится объект?

Клиент: Можно сначала поставить солнечную станцию на 60 кВт, а потом расширить?
Консультант: Да, станцию можно вводить поэтапно: начать с 60 кВт, а затем нарастить мощность. Возможность и параметры расширения лучше учесть уже при проектировании. У вас есть данные о дневном потреблении объекта?

Клиент: В счётчиках стоят симки другого оператора. Для вашей платформы их надо менять?
Консультант: Нет, менять SIM-карты только из-за оператора не обязательно. Наша платформа поддерживает передачу данных с приборов, использующих SIM-карты других операторов. Совместимость конкретных устройств и схемы сбора данных проверим отдельно. Какие у вас модели счётчиков?

Клиент: Нам предлагают компенсировать реактивную энергию. Сначала надо что-то измерить?
Консультант: Да, сначала нужно измерить параметры реактивной энергии и понять, есть ли избыточное потребление. Только после этого можно оценивать необходимость компенсирующих устройств. Без данных измерений рекомендовать конкретное оборудование было бы преждевременно.

(диалог) Клиент: Мы электросетевая компания. Хотим лучше находить потери. … Клиент: Архив примерно за два года, по одним приборам почасовые данные, по другим суточные. Таких данных хватит, чтобы начать?
Консультант: Да, двухлетнего архива достаточно по длительности: в наших материалах минимальная глубина — один год. Суточные данные подходят, если есть разбивка потребления на день и ночь; почасовые данные желательны для более точного анализа. Также потребуются обезличенные идентификаторы каналов учёта и сведения о приборах. Система сформирует список приоритетных проверок, но не установит факт хищения.

(диалог) Клиент: Мы СНТ, у нас 120 участков, счётчики не у всех. … Консультант: Как сейчас собираете показания? Клиент: Сами садоводы присылают цифры в чат. Что нам подойдёт?
Консультант: Вам подойдёт облачная платформа учёта: она ведёт учёт по каждому участку и местам общего пользования и позволяет дистанционно управлять нагрузкой. Показания с подключённых счётчиков собираются автоматически, без чатов и обходов. Сколько участков уже оснащены счётчиками и какие это модели?

(диалог) Клиент: Интересует выход на оптовый рынок. … Консультант: Можете оставить телефон для расчёта? Клиент: Пока нет, сначала хочу понять, подходим ли мы. Что для этого нужно?
Консультант: Конечно, контакты не обязательны. Главный параметр — присоединённая мощность: ниже 750 кВт выход на оптовый рынок невозможен, а от 750 кВт до 20 МВт участие идёт через энергосбытовую компанию-посредника. Также важны месячный объём и профиль потребления. Какая присоединённая мощность указана у вас в документах?

Клиент: Как настроить передачу данных со счётчика по RS-485?
Консультант: Настройка конкретных приборов и интерфейсов — вне компетенции консультанта, в наших материалах таких инструкций нет. Могу передать вопрос специалисту, который разберёт вашу схему подключения. Какие модели счётчиков у вас установлены?

Клиент: Если расчёт покажет, что проект нам невыгоден, вы так и скажете?
Консультант: Да, обязательно. Предварительная оценка экономического эффекта бесплатна и ни к чему вас не обязывает. Если расчёт покажет, что ожидаемая экономия недостаточна или проект не окупается, мы сообщим об этом, а не будем рекомендовать невыгодное решение."""


PRODUCT_DOC_RE = re.compile(r"(^|/)2[1-6]-[^/]+\.md$")


def build_context(kb: KnowledgeBase, hits: Sequence[Hit], max_chars: int = 11000, guard_chars: int = 5000) -> tuple[str, str]:
    """Возвращает (контекст найденных разделов, правила продуктов из выдачи)."""
    blocks, used = [], 0
    for h in hits:
        block = f"[{h.chunk.label}]\n{h.chunk.text}"
        if used + len(block) > max_chars and blocks:
            continue
        blocks.append(block)
        used += len(block)

    # Правила берём для продуктов из верхней части выдачи (не более двух).
    products: list[str] = []
    for h in hits[:5]:
        src = h.chunk.source
        if PRODUCT_DOC_RE.search(src) and src in kb.guards and src not in products:
            products.append(src)
    guard_blocks, gused = [], 0
    for src in products[:2]:
        title = next((c.title for c in kb.chunks if c.source == src), src)
        text = kb.guards[src]
        if gused + len(text) > guard_chars:
            text = text[: max(0, guard_chars - gused)].rsplit("\n", 1)[0]
        if text:
            guard_blocks.append(f"[{title}]\n{text}")
            gused += len(text)
    return "\n\n---\n\n".join(blocks), "\n\n---\n\n".join(guard_blocks)


def build_messages(dialog: Dialog, context: str, guards: str, *, extra: str = "") -> list[dict]:
    parts = [f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context or '(ничего релевантного не найдено)'}"]
    if guards:
        parts.append(f"ПРАВИЛА ПРОДУКТА (что можно и нельзя утверждать):\n{guards}")
    if dialog.is_multi_turn:
        parts.append(f"ИСТОРИЯ ДИАЛОГА:\n{dialog.history_text()}")
        parts.append(f"ПОСЛЕДНЯЯ РЕПЛИКА КЛИЕНТА (ответь на неё с учётом истории):\n{dialog.last_user}")
    else:
        parts.append(f"ВОПРОС КЛИЕНТА:\n{dialog.last_user}")
    if extra:
        parts.append(extra)
    parts.append(
        "Ответь как консультант МТС EnergyTool: прямой ответ на заданный вопрос, 1–2 нужных факта из контекста, "
        "при необходимости один уточняющий вопрос о следующем шаге. Без догадок и лишних подробностей. "
        "Обычный текст, 2–4 предложения, 35–70 слов."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def clean_answer(text: str, max_words: int = 170) -> str:
    text = strip_reasoning(text)
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    text = re.sub(r"^(?:Консультант|Ответ|Ассистент)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*|__|^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    text = re.sub(r"\[(?:[^\]]*\.md[^\]]*)\]", "", text)  # ссылки на файлы базы
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = []
    for ln in lines:  # строки бывшего списка склеиваем в предложения
        if joined and not re.search(r"[.!?:;…]$", joined[-1]):
            joined[-1] += ";" if ln[:1].islower() else "."
        joined.append(ln)
    text = re.sub(r"\s+", " ", " ".join(joined)).strip()
    words = text.split()
    if len(words) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept, count = [], 0
        for s in sentences:
            n = len(s.split())
            if kept and count + n > max_words:
                break
            kept.append(s)
            count += n
        # Если отрезали финальный уточняющий вопрос — возвращаем его.
        if sentences and sentences[-1].endswith("?") and sentences[-1] not in kept and len(sentences[-1].split()) < 25:
            kept.append(sentences[-1])
        text = " ".join(kept)
    return text[:9500]


NO_INFO = (
    "В материалах МТС EnergyTool нет точного ответа на этот вопрос, поэтому не буду его додумывать. "
    "Предлагаю передать вопрос специалисту: оставьте, пожалуйста, ФИО, телефон и название компании — "
    "с вами свяжутся и разберут ситуацию по вашему объекту."
)


def extractive_answer(dialog: Dialog, hits: Sequence[Hit]) -> str:
    """Безопасный ответ без LLM: несколько предложений из лучшего раздела."""
    if not hits:
        return NO_INFO
    q = set(expand(tokenize(dialog.last_user)))
    scored = []
    for rank, h in enumerate(hits[:3]):
        text = re.sub(r"\*\*([^*]+)\*\*\s*", "", h.chunk.text)  # жирные подзаголовки абзацев
        text = re.sub(r"\s+", " ", re.sub(r"(?m)^\s*[-*]\s+", "", text)).replace("--", "–")
        for s in re.split(r"(?<=[.!?])\s+", text):
            if 30 <= len(s) <= 400:
                overlap = len(q & set(tokenize(s)))
                scored.append((overlap - 0.3 * rank, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [s for score, s in scored[:2] if score > 0]
    if not picked:
        return NO_INFO
    body = " ".join(picked)
    return f"{body} Точные условия для вашего объекта уточнит специалист — могу передать заявку на бесплатную предварительную оценку."


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    mode: str  # llm | extractive | error
    latency_ms: int = 0
    error: str = ""

    def sources(self, limit: int = 6) -> list[dict]:
        return [
            {
                "file": h.chunk.source,
                "title": h.chunk.title,
                "heading": h.chunk.heading,
                "score": round(float(h.score), 4),
                "via": h.via,
                "excerpt": h.chunk.excerpt(),
            }
            for h in self.hits[:limit]
        ]


class Consultant:
    """Фасад: загрузить базу один раз, отвечать на вопросы потокобезопасно."""

    def __init__(self, kb_path: str | Path, settings: Settings | None = None, cache_dir: str | Path | None = None):
        self.s = settings or Settings.from_env()
        self.kb = KnowledgeBase.load(kb_path)
        self.client = ProxyClient(self.s) if self.s.llm_ready else None
        self.retriever = Retriever(self.kb, self.client, self.s, cache_dir=cache_dir)

    @property
    def llm_ready(self) -> bool:
        return self.client is not None

    def answer(self, dialog: Dialog, *, deadline: float | None = None, extra: str = "") -> Answer:
        started = time.perf_counter()
        hits: list[Hit] = []
        try:
            hits = self.retriever.search(dialog, deadline=deadline)
            if not self.client or _expired(deadline, 10):
                text, mode, err = extractive_answer(dialog, hits), "extractive", ""
            else:
                context, guards = build_context(self.kb, hits)
                messages = build_messages(dialog, context, guards, extra=extra)
                timeout, retries = None, None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    timeout = max(10.0, min(self.s.timeout, remaining - 5))
                    retries = 3 if remaining > 3 * self.s.timeout else 1
                try:
                    text = clean_answer(self.client.chat(messages, timeout=timeout, retries=retries))
                    if len(text.split()) < 5:
                        raise LLMError("слишком короткий ответ")
                    if self.s.self_check and not _expired(deadline, 3 * self.s.timeout):
                        text = self._self_check(dialog, context, guards, text, timeout, retries)
                    mode, err = "llm", ""
                except Exception as e:  # noqa: BLE001
                    text, mode, err = extractive_answer(dialog, hits), "extractive", f"{type(e).__name__}: {e}"[:300]
        except Exception as e:  # noqa: BLE001
            text, mode, err = NO_INFO, "error", f"{type(e).__name__}: {e}"[:300]
        return Answer(text or NO_INFO, hits, mode, round((time.perf_counter() - started) * 1000), err)

    def _self_check(self, dialog: Dialog, context: str, guards: str, draft: str, timeout, retries) -> str:
        """Второй проход: убрать из черновика всё, что не подтверждено контекстом."""
        prompt = (
            f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}\n\n"
            + (f"ПРАВИЛА ПРОДУКТА:\n{guards}\n\n" if guards else "")
            + (f"ИСТОРИЯ ДИАЛОГА:\n{dialog.history_text()}\n\n" if dialog.is_multi_turn else "")
            + f"ВОПРОС КЛИЕНТА:\n{dialog.last_user}\n\nЧЕРНОВИК ОТВЕТА:\n{draft}\n\n"
            "Проверь черновик по контексту. Исправь или удали утверждения, которых нет в контексте или "
            "которые противоречат ему и правилам продукта; верни недостающий ключевой факт, если он прямо "
            "отвечает на вопрос. Сохрани стиль: прямой ответ, суть, при необходимости один уточняющий вопрос, "
            "обычно 40–90 слов, обычный текст. Если черновик верен — верни его без изменений. "
            "Выведи только итоговый ответ, без пояснений."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        try:
            checked = clean_answer(self.client.chat(messages, timeout=timeout, retries=retries, temperature=0.0))
            return checked if len(checked.split()) >= 5 else draft
        except Exception:  # noqa: BLE001
            return draft

    def complete(self, messages: list[dict], *, max_tokens: int = 600) -> str | None:
        """Свободная генерация (например, резюме заявки). None — если LLM недоступна."""
        if not self.client:
            return None
        try:
            return strip_reasoning(self.client.chat(messages, max_tokens=max_tokens, temperature=0.1))
        except Exception:  # noqa: BLE001
            return None
