"""RAG для консультанта EnergyTool: база знаний, поиск, генерация ответа."""

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


def env(*names: str, default: str = "") -> str:
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
    temperature: float = 0.0
    max_tokens: int = 700
    top_k: int = 8
    rerank_candidates: int = 24
    timeout: float = 60.0
    use_embeddings: bool = True
    use_reranker: bool = True
    emb_base_url: str = ""
    emb_api_key: str = ""
    enable_thinking: bool = False
    self_check: bool = False
    fewshot_k: int = 3
    rerank_template: bool = True
    fewshot_exclude_exact: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        max_tokens = env_number("MAX_TOKENS", 2048, 64, 8192)
        return cls(
            base_url=env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/"),
            api_key=env("LLM_API_KEY", "OPENAI_API_KEY"),
            model=env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default="qwen/qwen3.8-27b"),
            emb_model=env("EMB_MODEL", default="BAAI/bge-m3"),
            reranker_model=env("RERANKER_MODEL", default="Qwen/Qwen3-Reranker-8B"),

            temperature=env_number("ANSWER_TEMPERATURE", 0.0, 0.0, 1.0, float),  # для фактов нужна детерминированность

            # если прокси проигнорирует отключение размышлений, они тоже съедают max_tokens
            max_tokens=max(2048, min(max_tokens, 6000)),
            top_k=max(6, env_number("TOP_K", 8, 3, 16)),
            rerank_candidates=max(env_number("RERANK_CANDIDATES", 24, 4, 40), 8),
            timeout=float(env_number("LLM_TIMEOUT", 60, 10, 180)),
            use_embeddings=env_flag("USE_EMBEDDINGS", True),
            use_reranker=env_flag("USE_RERANKER", True),
            emb_base_url=env("EMB_BASE_URL").rstrip("/"),
            emb_api_key=env("EMB_API_KEY"),
            enable_thinking=env_flag("ENABLE_THINKING", True),
            self_check=env_flag("SELF_CHECK", False),
            fewshot_k=env_number("FEWSHOT_K", 4, 0, 8),
            rerank_template=env_flag("RERANK_TEMPLATE", True),
            fewshot_exclude_exact=env_flag("FEWSHOT_EXCLUDE_EXACT", False),
        )

    @property
    def llm_ready(self) -> bool:
        return bool(self.base_url and self.api_key)


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


# как говорят клиенты -> как написано в базе
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


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    heading: str
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
        while len(block) > max_chars:
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


# разделы с ограничениями по продукту, их всегда кладём в промпт
GUARD_HEADINGS = ("факты, которые можно утверждать", "чего утверждать нельзя", "что мы не делаем")


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    guards: dict[str, str]
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
        # синонимы с половинным весом
        terms = tokenize(query)
        weights = Counter(terms)
        for t in expand(terms):
            weights.setdefault(t, 0.5)
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


class LLMError(RuntimeError):
    pass


class ProxyClient:
    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, settings: Settings, retries: int = 3):
        self.s = settings
        self.retries = retries
        self.http = httpx.Client(
            timeout=httpx.Timeout(settings.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        self.no_think_kwargs = True  # сбросим, если прокси не знает chat_template_kwargs
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
            except ValueError as e:
                last = f"bad json: {e}"
        raise LLMError(f"{endpoint}: {last}")

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
        if self.s.rerank_template:
            query = RERANK_PREFIX + f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n"
            documents = [f"<Document>: {d}{RERANK_SUFFIX}" for d in documents]
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

            status = str(e)[:8]
            if sent_kwargs and status.startswith("HTTP 4") and status not in ("HTTP 401", "HTTP 403"):
                with self._lock:
                    self.no_think_kwargs = False
                payload.pop("chat_template_kwargs", None)
                payload["messages"] = _no_think(messages, self.s.enable_thinking)
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
            finish = (data["choices"][0].get("finish_reason") or "")
            if finish == "length" and payload["max_tokens"] < 8000:
                # модель всё-таки размышляла и не успела ответить: даём больше токенов один раз
                payload["max_tokens"] = min(8000, payload["max_tokens"] * 3)
                payload["messages"] = _no_think(messages, self.s.enable_thinking)
                data = self.post("/chat/completions", payload, timeout=timeout, retries=retries)
                content = strip_reasoning(str(data["choices"][0]["message"].get("content") or ""))
            if not content.strip():
                raise LLMError("chat: пустой ответ (возможно, токены ушли на размышления)")
        return content


def _no_think(messages: list[dict], thinking: bool) -> list[dict]:
    # мягкий выключатель размышлений qwen3, если прокси не понял chat_template_kwargs
    if thinking:
        return messages
    msgs = [dict(m) for m in messages]
    msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n/no_think"
    return msgs


# формат запроса, рекомендованный для Qwen3-Reranker в /v1/rerank (документация vLLM)
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_INSTRUCTION = (
    "Given a customer's question to an energy services consultant, retrieve knowledge base passages "
    "that contain facts needed to answer it"
)


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def strip_reasoning(text: str) -> str:
    # qwen3 может вернуть <think>...</think> прямо в content
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


TURN_RE = re.compile(
    r"^\s*(пользователь|клиент|user|ассистент|консультант|бот|assistant)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Dialog:
    turns: list[tuple[str, str]]

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
    if turns[-1][0] != "user":
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
        except Exception as e:
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
        sims = [_dot(qvec, v) for v in self.vectors]
        return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:n]

    def search(self, dialog: Dialog, *, deadline: float | None = None, fast: bool = False) -> list[Hit]:
        last = dialog.last_user
        users = dialog.user_turns

        context_parts = [last] + users[-3:-1][::-1]
        if dialog.is_multi_turn:
            prev_bot = next((t for r, t in reversed(dialog.turns[:-1]) if r == "assistant"), "")
            if prev_bot:
                context_parts.append(prev_bot)
        context_query = " ".join(context_parts)

        rankings: list[tuple[list[int], float]] = [(self._rank_bm25(last, 40), 1.0)]
        if dialog.is_multi_turn:
            rankings.append((self._rank_bm25(context_query, 40), 0.7))

        if self.vectors is not None and self.client and not fast and not _expired(deadline, 20):
            try:
                queries = [last] + ([context_query] if dialog.is_multi_turn else [])
                qvecs = self.client.embed(queries)
                rankings.append((self._rank_dense(qvecs[0], 40), 1.3))
                if len(qvecs) > 1:
                    rankings.append((self._rank_dense(qvecs[1], 40), 1.0))
            except Exception:
                pass

        # reciprocal rank fusion
        fused: dict[int, float] = defaultdict(float)
        for ranking, weight in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] += weight / (60 + rank)
        order = sorted(fused, key=fused.get, reverse=True)
        candidates = order[: self.s.rerank_candidates]
        hits = [Hit(self.kb.chunks[i], fused[i], "rrf") for i in candidates]

        if self.rerank_ok and self.client and self.s.llm_ready and len(hits) > 2 and not fast and not _expired(deadline, 25):
            rq = last if not dialog.is_multi_turn else f"{last}\nКонтекст диалога: {context_query}"
            try:
                ranked = self.client.rerank(rq, [h.chunk.for_embedding()[:800] for h in hits], top_n=len(hits))
                # реранкер не заменяет порядок целиком, а голосует вместе с гибридным поиском
                score = {i: 1.0 / (10 + i) for i in range(len(hits))}
                for r, (i, _s) in enumerate(ranked):
                    score[i] += 1.5 / (10 + r)
                order = sorted(score, key=score.get, reverse=True)
                return [Hit(hits[i].chunk, score[i], "rerank") for i in order[: self.s.top_k]]
            except Exception:
                pass
        return hits[: self.s.top_k]


def _expired(deadline: float | None, reserve: float = 0.0) -> bool:
    return deadline is not None and time.monotonic() > deadline - reserve


SYSTEM_PROMPT = """Ты — ИИ-консультант МТС EnergyTool на сайте компании. К тебе приходят представители бизнеса, энергетики предприятий, председатели СНТ, сетевые организации и менеджеры МТС. Они пишут своими словами и часто не знают терминов.

Как ты отвечаешь:
1. Первое предложение — прямой ответ именно на заданный вопрос («Да, …», «Нет, …», «Да, но …», «Может иметь смысл, но …», «Начать стоит с …»). Не пересказывай вопрос и не здоровайся.
2. Затем 1–2 предложения по существу: только те факты из базы знаний, которые нужны для ответа на этот вопрос, и главная оговорка (от чего зависит результат или что нужно проверить). Не перечисляй всё, что знаешь о продукте. Не называй цифры, которые не отвечают на вопрос (точность распознавания, число поддерживаемых приборов, проценты, сроки и т. п.), — их называй, только если клиент спросил именно об этом.
3. В конце — один уточняющий вопрос о главном недостающем параметре для расчёта или подбора решения. Выбирай его из блока «ЧТО УТОЧНЯТЬ У КЛИЕНТА» (регион, присоединённая мощность, объём и профиль потребления, количество приборов и способ сбора показаний, ценовая категория, тип кровли и т. п.). Не спрашивай технические мелочи (тип индикатора, ориентацию стен, модель инвертора) и то, что клиент уже сообщил. Если вопрос справочный и полностью закрыт ответом, уточняющий вопрос не нужен.
4. Объём — 2–4 предложения, обычно 35–70 слов. Обычный текст одним абзацем: без списков, заголовков, Markdown и ссылок на разделы базы.
5. Говори от лица МТС EnergyTool («мы», «наша платформа»).

Рассуждай как консультант, а не как поисковик:
- Свяжи факты базы с ситуацией клиента из диалога. Например, сетевая солнечная станция вырабатывает энергию днём и без накопителей не переносит её на вечер — значит, при вечернем потреблении эффект ниже и стоит сравнить с гибридной станцией.
- Не строй догадок о причинах ситуации клиента («скорее всего, у вас …»). Вместо этого скажи, от чего это может зависеть и что сравнить или проверить.
- Если в диалоге клиент уже ответил на вопрос консультанта, используй это и двигайся дальше.
- Говори о продукте, о котором спрашивает клиент. Другой продукт упоминай, только если он прямо решает задачу клиента (например, распознавание по фото для приборов, которые нельзя подключить к платформе), и не предлагай его «на всякий случай».

Правила достоверности:
- Факты о продуктах, ценах, сроках, тарифах, порогах, законах и условиях бери ТОЛЬКО из КОНТЕКСТА. Не добавляй знаний «от себя», не округляй и не меняй цифры.
- Если в контексте нет ответа или вопрос не про продукты EnergyTool и связанные задачи клиента, прямо скажи, что в наших материалах этого нет / это вне компетенции консультанта, и предложи связаться со специалистом. Не выдумывай.
- Соблюдай раздел «ПРАВИЛА ПРОДУКТА»: не утверждай то, что там запрещено (итоговая стоимость, гарантированная экономия, точный срок окупаемости до расчёта, имена клиентов и производителей, внутренние алгоритмы, автономность сетевой станции и т. п.).
- Индикативные цифры называй только с оговоркой, что точный результат — после расчёта по объекту.
- Контакты (ФИО, телефон, компания) предлагай оставить, только если клиент сам хочет расчёт, звонок или специалиста. Если клиент сказал, что не готов оставлять контакты, — не проси их, просто помоги по существу.
- Технические инструкции по настройке приборов и юридические заключения — вне компетенции: предложи специалиста.

Примеры хороших ответов:

(диалог) Клиент: Мы СНТ, у нас 120 участков, счётчики не у всех. … Консультант: Как сейчас собираете показания? Клиент: Сами садоводы присылают цифры в чат. Что нам подойдёт?
Консультант: Вам подойдёт облачная платформа учёта: она ведёт учёт по каждому участку и местам общего пользования и позволяет дистанционно управлять нагрузкой. Показания с подключённых счётчиков собираются автоматически, без чатов и обходов. Сколько участков уже оснащены счётчиками и какие это модели?

(диалог) Клиент: Интересует выход на оптовый рынок. … Консультант: Можете оставить телефон для расчёта? Клиент: Пока нет, сначала хочу понять, подходим ли мы. Что для этого нужно?
Консультант: Конечно, контакты не обязательны. Главный параметр — присоединённая мощность: ниже 750 кВт выход на оптовый рынок невозможен, а от 750 кВт до 20 МВт участие идёт через энергосбытовую компанию-посредника. Также важны месячный объём и профиль потребления. Какая присоединённая мощность указана у вас в документах?

Клиент: Как настроить передачу данных со счётчика по RS-485?
Консультант: Настройка конкретных приборов и интерфейсов — вне компетенции консультанта, в наших материалах таких инструкций нет. Могу передать вопрос специалисту, который разберёт вашу схему подключения. Какие модели счётчиков у вас установлены?"""


class ExampleBank:
    def __init__(self, pairs: Sequence[tuple[str, str]]):
        self.pairs = list(pairs)
        self.items = []
        for q, _ in self.pairs:
            d = parse_dialog(q)
            text = f"{d.last_user} {d.last_user} " + " ".join(d.user_turns[:-1])
            self.items.append(Chunk("", "", "", "", text, tokenize(text)))
        self.bm25 = BM25(self.items) if self.items else None
        self.keys = [normalize(q) for q, _ in self.pairs]

    @classmethod
    def load(cls) -> "ExampleBank":
        try:
            from examples import EXAMPLES
        except ImportError:
            EXAMPLES = []
        return cls(EXAMPLES)

    def pick(self, dialog: Dialog, k: int, exclude_exact: bool = False) -> list[tuple[str, str]]:
        if not self.bm25 or k <= 0:
            return []
        query = f"{dialog.last_user} {dialog.last_user} " + " ".join(dialog.user_turns[:-1])
        scores = self.bm25.scores(query)
        key = normalize(" ".join(t for _, t in dialog.turns))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if exclude_exact and (self.keys[i] == key or normalize(parse_dialog(self.pairs[i][0]).last_user) == normalize(dialog.last_user)):
                continue
            out.append(self.pairs[i])
            if len(out) >= k:
                break
        return out


def format_examples(pairs: Sequence[tuple[str, str]]) -> str:
    blocks = []
    for q, a in pairs:
        d = parse_dialog(q)
        if d.is_multi_turn:
            q_text = "(диалог) " + " … ".join(f"{'Клиент' if r == 'user' else 'Консультант'}: {t}" for r, t in d.turns)
        else:
            q_text = f"Клиент: {d.last_user}"
        blocks.append(f"{q_text}\nКонсультант: {a}")
    return "\n\n".join(blocks)


PRODUCT_DOC_RE = re.compile(r"(^|/)2[1-6]-[^/]+\.md$")


def build_context(kb: KnowledgeBase, hits: Sequence[Hit], max_chars: int = 11000, guard_chars: int = 5000) -> tuple[str, str]:
    blocks, used = [], 0
    for h in hits:
        block = f"[{h.chunk.label}]\n{h.chunk.text}"
        if used + len(block) > max_chars and blocks:
            continue
        blocks.append(block)
        used += len(block)

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
    ask = _ask_block(kb, hits)
    if ask:
        guard_blocks.append(ask)
    return "\n\n---\n\n".join(blocks), "\n\n---\n\n".join(guard_blocks)


ASK_SECTIONS = {
    "21-orem": "Для анализа затрат и выхода на оптовый рынок",
    "11-price": "Для анализа затрат и выхода на оптовый рынок",
    "14-tariffs": "Для анализа затрат и выхода на оптовый рынок",
    "23-solar": "Для солнечной станции",
    "15-micro": "Для солнечной станции",
    "22-askue": "Для платформы учёта",
    "24-losses": "Для поиска коммерческих потерь",
    "25-meter": "Для распознавания показаний",
}


def _ask_block(kb: KnowledgeBase, hits: Sequence[Hit]) -> str:
    wanted: list[str] = []
    for h in hits[:4]:
        for prefix, section in ASK_SECTIONS.items():
            if prefix in h.chunk.source and section not in wanted:
                wanted.append(section)
    wanted = wanted[:2] or ["Для первичного понимания задачи"]
    parts = []
    for c in kb.chunks:
        if c.source.endswith("03-calculation-and-requests.md") and c.heading.split(" → ")[-1] in wanted:
            parts.append(f"{c.heading.split(' → ')[-1]}:\n{c.text}")
    if not parts:
        return ""
    return "ЧТО УТОЧНЯТЬ У КЛИЕНТА (уточняющий вопрос выбирай отсюда, о самом важном недостающем параметре):\n" + "\n\n".join(parts)


def build_messages(dialog: Dialog, context: str, guards: str, *, extra: str = "", examples: str = "") -> list[dict]:
    parts = []
    if examples:
        parts.append(
            "ПРИМЕРЫ ОТВЕТОВ НА ПОХОЖИЕ ВОПРОСЫ (ориентир по стилю, длине и уровню детализации; "
            f"факты для ответа бери только из контекста ниже):\n{examples}"
        )
    parts.append(f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context or '(ничего релевантного не найдено)'}")
    if guards:
        parts.append(f"ПРАВИЛА ПРОДУКТА И ЧТО УТОЧНЯТЬ:\n{guards}")
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
        "Структура ответа: назови продукт, приведи 2–3 ключевых факта из контекста, закончи одним уточняющим вопросом. "
        "Отвечай кратко: 1–2 предложения, 35–70 слов."
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
    text = re.sub(r"\[(?:[^\]]*\.md[^\]]*)\]", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = []
    for ln in lines:
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
    if not hits:
        return NO_INFO
    q = set(expand(tokenize(dialog.last_user)))
    scored = []
    for rank, h in enumerate(hits[:3]):
        text = re.sub(r"\*\*([^*]+)\*\*\s*", "", h.chunk.text)
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
    mode: str
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
    def __init__(self, kb_path: str | Path, settings: Settings | None = None, cache_dir: str | Path | None = None):
        self.s = settings or Settings.from_env()
        self.kb = KnowledgeBase.load(kb_path)
        self.client = ProxyClient(self.s) if self.s.llm_ready else None
        self.retriever = Retriever(self.kb, self.client, self.s, cache_dir=cache_dir)
        self.examples = ExampleBank.load()

    @property
    def llm_ready(self) -> bool:
        return self.client is not None

    def answer(self, dialog: Dialog, *, deadline: float | None = None, extra: str = "", fast: bool = False) -> Answer:
        started = time.perf_counter()
        hits: list[Hit] = []
        try:
            hits = self.retriever.search(dialog, deadline=deadline, fast=fast)
            if not self.client or _expired(deadline, 10):
                text, mode, err = extractive_answer(dialog, hits), "extractive", ""
            else:
                context, guards = build_context(self.kb, hits)
                examples = format_examples(self.examples.pick(dialog, self.s.fewshot_k, self.s.fewshot_exclude_exact))
                messages = build_messages(dialog, context, guards, extra=extra, examples=examples)
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
                except Exception as e:
                    text, mode, err = extractive_answer(dialog, hits), "extractive", f"{type(e).__name__}: {e}"[:300]
        except Exception as e:
            text, mode, err = NO_INFO, "error", f"{type(e).__name__}: {e}"[:300]
        return Answer(text or NO_INFO, hits, mode, round((time.perf_counter() - started) * 1000), err)

    def _self_check(self, dialog: Dialog, context: str, guards: str, draft: str, timeout, retries) -> str:
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
        except Exception:
            return draft

    def complete(self, messages: list[dict], *, max_tokens: int = 600) -> str | None:
        if not self.client:
            return None
        try:
            return strip_reasoning(self.client.chat(messages, max_tokens=max_tokens, temperature=0.1))
        except Exception:
            return None
