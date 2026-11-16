/* ==========================================================================
   МТС EnergyTool — клиент ИИ-консультанта
   Без фреймворков: DOM, fetch, localStorage.
   ========================================================================== */

(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const els = {
    body: $("#chatBody"),
    form: $("#chatForm"),
    input: $("#chatInput"),
    send: $("#sendBtn"),
    quick: $("#quickReplies"),
    debugToggle: $("#debugToggle"),
    debugPanel: $("#debugPanel"),
    dbgMode: $("#dbgMode"),
    dbgLatency: $("#dbgLatency"),
    dbgIntent: $("#dbgIntent"),
    dbgSlots: $("#dbgSlots"),
    dbgModel: $("#dbgModel"),
    historyToggle: $("#historyToggle"),
    historyDrawer: $("#historyDrawer"),
    historyList: $("#historyList"),
    historyClose: $("#historyClose"),
    clearHistory: $("#clearHistory"),
    newDialog: $("#newDialog"),
    authToggle: $("#authToggle"),
    authModal: $("#authModal"),
    authClose: $("#authClose"),
    authForm: $("#authForm"),
    authSubmit: $("#authSubmit"),
    authSwitch: $("#authSwitch"),
    authTitle: $("#authTitle"),
    authError: $("#authError"),
    authNameField: $("#authNameField"),
    authPhoneField: $("#authPhoneField"),
    authCompanyField: $("#authCompanyField"),
    cabinetModal: $("#cabinetModal"),
    cabinetClose: $("#cabinetClose"),
    cabinetLogout: $("#cabinetLogout"),
    cabinetBody: $("#cabinetBody"),
    cabinetTitle: $("#cabinetTitle"),
    toast: $("#toast"),
    caption: $(".chat-caption"),
  };

  const state = {
    sessionId: localStorage.getItem("et_session_id") || null,
    debug: false,
    busy: false,
    leadDone: false,
    lastDebug: null,
    token: localStorage.getItem("et_token") || null,
    user: null,
    authMode: "login",
  };

  const INITIAL_QUICK = [
    "Снизить счёт за электричество",
    "Наладить учёт",
    "Найти потери",
    "Выйти на оптовый рынок",
  ];

  let toastTimer = null;

  // ------------------------------------------------------------------
  // Утилиты
  // ------------------------------------------------------------------

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function svgIcon(id, className = "icon") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", className);
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${id}`);
    svg.appendChild(use);
    return svg;
  }

  function timeLabel(value) {
    const date = value ? new Date(value) : new Date();
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
  }

  function toast(message, isError = false) {
    els.toast.textContent = message;
    els.toast.classList.toggle("is-error", isError);
    els.toast.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove("is-visible"), 3600);
  }

  function scrollBottom(smooth = true) {
    requestAnimationFrame(() => {
      els.body.scrollTo({ top: els.body.scrollHeight, behavior: smooth ? "smooth" : "auto" });
    });
  }

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (state.token) headers["Authorization"] = "Bearer " + state.token;
    const response = await fetch(path, {
      headers,
      ...options,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) {
      const detail = data && data.detail;
      let message = "Что-то пошло не так. Попробуйте ещё раз.";
      if (typeof detail === "string") message = detail;
      else if (Array.isArray(detail) && detail.length) message = detail[0].msg || message;
      throw new Error(message);
    }
    return data;
  }

  // ------------------------------------------------------------------
  // Сообщения
  // ------------------------------------------------------------------

  function addMessage(role, text, meta = {}) {
    const wrapper = node("div", `msg msg-${role}`);

    if (role === "bot") {
      const avatar = node("div", "msg-avatar");
      avatar.appendChild(svgIcon("i-spark"));
      wrapper.appendChild(avatar);
    }

    const content = node("div", "msg-content");
    const bubble = node("div", "bubble", text);
    content.appendChild(bubble);

    const metaRow = node("div", "msg-meta");
    metaRow.appendChild(node("span", "msg-time", timeLabel(meta.created_at)));

    if (meta.latency_ms !== undefined && meta.latency_ms !== null) {
      metaRow.appendChild(node("span", "", `${meta.latency_ms} мс`));
    }

    if (Array.isArray(meta.sources) && meta.sources.length) {
      const toggle = node("button", "source-toggle");
      toggle.type = "button";
      toggle.appendChild(svgIcon("i-doc"));
      toggle.appendChild(node("span", "", `Источники · ${meta.sources.length}`));
      metaRow.appendChild(toggle);

      const box = node("div", "sources-box");
      meta.sources.forEach((source) => {
        const item = node("div", "source-item");
        const head = node("div", "source-head");
        head.appendChild(node("span", "", source.heading || source.title || source.file));
        head.appendChild(node("span", "source-score", Number(source.score || 0).toFixed(3)));
        item.appendChild(head);
        item.appendChild(node("div", "source-file", source.file || ""));
        if (source.excerpt) item.appendChild(node("div", "source-excerpt", source.excerpt));
        box.appendChild(item);
      });

      if (state.debug) box.classList.add("is-open");
      toggle.addEventListener("click", () => {
        box.classList.toggle("is-open");
      });

      content.appendChild(metaRow);
      content.appendChild(box);
    } else {
      content.appendChild(metaRow);
    }

    wrapper.appendChild(content);
    els.body.appendChild(wrapper);
    scrollBottom();
    return wrapper;
  }

  function showTyping() {
    const wrapper = node("div", "msg msg-bot");
    wrapper.id = "typingIndicator";
    const avatar = node("div", "msg-avatar");
    avatar.appendChild(svgIcon("i-spark"));
    wrapper.appendChild(avatar);

    const content = node("div", "msg-content");
    const bubble = node("div", "bubble");
    const typing = node("span", "typing");
    typing.appendChild(node("i"));
    typing.appendChild(node("i"));
    typing.appendChild(node("i"));
    bubble.appendChild(typing);
    content.appendChild(bubble);
    wrapper.appendChild(content);
    els.body.appendChild(wrapper);
    scrollBottom();
  }

  function hideTyping() {
    const indicator = $("#typingIndicator");
    if (indicator) indicator.remove();
  }

  function renderQuickReplies(items) {
    els.quick.innerHTML = "";
    (items || []).forEach((text) => {
      const chip = node("button", "quick-chip", text);
      chip.type = "button";
      chip.addEventListener("click", () => {
        if (!state.busy) sendMessage(text);
      });
      els.quick.appendChild(chip);
    });
  }

  function renderWelcome() {
    els.body.innerHTML = "";
    els.body.appendChild(node("div", "chat-day", "Сегодня"));
    addMessage(
      "bot",
      "Здравствуйте! Я ИИ-консультант МТС EnergyTool.\n\nПомогу разобраться с учётом электроэнергии, солнечной станцией, поиском потерь или выходом на оптовый рынок. Опишите задачу своими словами — задам пару уточняющих вопросов и предложу подходящее решение."
    );
    renderQuickReplies(INITIAL_QUICK);
  }

  // ------------------------------------------------------------------
  // Отправка сообщения
  // ------------------------------------------------------------------

  async function sendMessage(rawText) {
    const text = String(rawText || "").trim();
    if (!text || state.busy) return;

    state.busy = true;
    els.send.disabled = true;
    els.input.value = "";
    resizeInput();
    renderQuickReplies([]);
    addMessage("user", text);
    showTyping();

    try {
      const data = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ session_id: state.sessionId, message: text }),
      });

      hideTyping();
      state.sessionId = data.session_id;
      localStorage.setItem("et_session_id", state.sessionId);
      state.lastDebug = data.debug || null;
      state.leadDone = Boolean(data.lead && data.lead.done);

      addMessage("bot", data.answer, {
        sources: data.sources,
        latency_ms: data.debug ? data.debug.latency_ms : null,
      });

      renderQuickReplies(data.quick_replies || []);

      if (state.debug) {
        updateDebugPanel(data.debug);
        expandAllSources(true);
      }

      if (data.lead && data.lead.offered && !data.lead.done && !state.leadDone) {
        appendLeadForm(data.lead.prefill || {});
      }
    } catch (error) {
      hideTyping();
      addMessage("bot", `Не удалось получить ответ: ${error.message}. Проверьте, что сервер запущен, и попробуйте ещё раз.`);
      renderQuickReplies(INITIAL_QUICK);
      toast(error.message, true);
    } finally {
      state.busy = false;
      els.send.disabled = false;
      els.input.focus();
      scrollBottom();
    }
  }

  // ------------------------------------------------------------------
  // Заявка
  // ------------------------------------------------------------------

  function appendLeadForm(prefill) {
    if ($("#leadFormCard")) return;

    const wrapper = node("div", "msg msg-bot");
    const avatar = node("div", "msg-avatar");
    avatar.appendChild(svgIcon("i-spark"));
    wrapper.appendChild(avatar);

    const content = node("div", "msg-content");
    const card = node("div", "lead-card");
    card.id = "leadFormCard";

    const title = node("div", "lead-title");
    title.appendChild(svgIcon("i-doc"));
    title.appendChild(node("span", "", "Передать заявку инженеру"));
    card.appendChild(title);
    card.appendChild(
      node(
        "div",
        "lead-sub",
        "Оставьте контакты — менеджер получит резюме диалога и подготовит бесплатный расчёт. В демо используйте тестовые данные."
      )
    );

    const form = node("form", "lead-form");
    const fields = [
      { name: "name", placeholder: "ФИО", type: "text", value: prefill.name || "" },
      { name: "phone", placeholder: "+7 999 123-45-67", type: "tel", value: prefill.phone || "" },
      { name: "company", placeholder: "Название компании", type: "text", value: prefill.company || "" },
    ];

    fields.forEach((field) => {
      const row = node("div", "lead-field");
      const input = document.createElement("input");
      input.name = field.name;
      input.type = field.type;
      input.placeholder = field.placeholder;
      input.value = field.value;
      input.required = true;
      input.autocomplete = field.name === "phone" ? "tel" : "off";
      if (field.name === "phone") {
        input.addEventListener("input", () => {
          input.value = formatPhone(input.value);
        });
      }
      row.appendChild(input);
      form.appendChild(row);
    });

    const consent = node("label", "lead-consent");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "consent";
    checkbox.required = true;
    consent.appendChild(checkbox);
    consent.appendChild(
      node("span", "", "Согласен на обработку персональных данных. Это демонстрационный прототип — используйте только тестовые данные.")
    );
    form.appendChild(consent);

    const submit = node("button", "lead-submit", "Отправить заявку");
    submit.type = "submit";
    form.appendChild(submit);
    card.appendChild(form);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submit.disabled = true;
      submit.textContent = "Отправляем…";
      try {
        const payload = {
          session_id: state.sessionId,
          name: form.elements.name.value.trim(),
          phone: form.elements.phone.value.trim(),
          company: form.elements.company.value.trim(),
          consent: form.elements.consent.checked,
        };
        const result = await api("/api/lead", { method: "POST", body: JSON.stringify(payload) });
        state.leadDone = true;
        card.replaceWith(leadSuccess(result));
        renderQuickReplies(["Продолжить консультацию", "Спасибо"]);
        scrollBottom();
      } catch (error) {
        submit.disabled = false;
        submit.textContent = "Отправить заявку";
        toast(error.message, true);
      }
    });

    content.appendChild(card);
    wrapper.appendChild(content);
    els.body.appendChild(wrapper);
    scrollBottom();
  }

  function leadSuccess(result) {
    const card = node("div", "lead-card");
    const success = node("div", "lead-success");
    success.appendChild(svgIcon("i-check"));
    const text = node("div");
    text.appendChild(node("strong", "", `Заявка №${result.lead_id} сформирована.`));
    text.appendChild(
      node("div", "", "Менеджер получил резюме диалога и свяжется с вами. Ниже — что именно уйдёт в заявку.")
    );
    success.appendChild(text);
    card.appendChild(success);

    const details = node("details", "summary-details");
    details.appendChild(node("summary", "", "Показать резюме для менеджера"));
    const pre = node("pre", "", result.summary || "");
    details.appendChild(pre);
    card.appendChild(details);
    return card;
  }

  function formatPhone(value) {
    let digits = value.replace(/\D/g, "");
    if (!digits) return "";
    if (digits[0] === "8") digits = "7" + digits.slice(1);
    if (digits[0] !== "7") digits = "7" + digits;
    digits = digits.slice(0, 11);
    const parts = ["+7"];
    if (digits.length > 1) parts.push(" (" + digits.slice(1, 4));
    if (digits.length >= 5) parts.push(") " + digits.slice(4, 7));
    if (digits.length >= 8) parts.push("-" + digits.slice(7, 9));
    if (digits.length >= 10) parts.push("-" + digits.slice(9, 11));
    return parts.join("");
  }

  // ------------------------------------------------------------------
  // Отладка
  // ------------------------------------------------------------------

  function updateDebugPanel(debug) {
    if (!debug) {
      els.dbgMode.textContent = "ожидание запроса";
      els.dbgLatency.textContent = "—";
      els.dbgIntent.textContent = "—";
      els.dbgSlots.textContent = "—";
      els.dbgModel.textContent = "—";
      return;
    }
    els.dbgMode.textContent = debug.mode === "llm" ? "LLM" : debug.mode === "local-fallback" ? "локальный fallback" : "локальный RAG";
    els.dbgLatency.textContent = `${debug.latency_ms} мс`;
    els.dbgIntent.textContent = `${debug.intent_label || "—"} · ${debug.product || "—"}`;
    const slots = debug.slots || {};
    const entries = Object.entries(slots).filter(([, value]) => value !== null && value !== "" && value !== undefined);
    els.dbgSlots.textContent = entries.length
      ? entries.map(([key, value]) => `${key}=${Array.isArray(value) ? value.join("/") : value}`).join(" · ")
      : "пока не собраны";
    const retrieval = debug.retrieval ? ` · поиск: ${debug.retrieval}` : "";
    els.dbgModel.textContent = (debug.llm_model || "демо-режим без LLM") + retrieval;
  }

  function expandAllSources(open) {
    $$(".sources-box", els.body).forEach((box) => box.classList.toggle("is-open", open));
  }

  function toggleDebug() {
    state.debug = !state.debug;
    els.debugToggle.classList.toggle("is-active", state.debug);
    els.debugToggle.setAttribute("aria-pressed", String(state.debug));
    els.debugPanel.hidden = !state.debug;
    expandAllSources(state.debug);
    if (state.debug) {
      if (state.lastDebug) updateDebugPanel(state.lastDebug);
      else {
        els.dbgMode.textContent = "ожидание запроса";
        els.dbgLatency.textContent = "—";
        els.dbgIntent.textContent = "—";
        els.dbgSlots.textContent = "—";
        els.dbgModel.textContent = "—";
      }
    }
    toast(state.debug ? "Режим отладки включён: видны источники, время ответа и слоты." : "Режим отладки выключен.");
  }

  // ------------------------------------------------------------------
  // История диалогов
  // ------------------------------------------------------------------

  async function openHistory() {
    els.historyDrawer.hidden = false;
    els.historyList.innerHTML = '<div class="history-empty">Загружаем историю…</div>';
    try {
      const data = await api("/api/sessions");
      const sessions = data.sessions || [];
      els.historyList.innerHTML = "";
      if (!sessions.length) {
        els.historyList.appendChild(node("div", "history-empty", "Пока нет сохранённых диалогов."));
        return;
      }
      sessions.forEach((session) => {
        const item = node("button", "history-item");
        item.type = "button";
        item.appendChild(node("strong", "", session.title || "Новый диалог"));
        const row = node("span");
        const date = new Date(session.updated_at);
        row.appendChild(
          node(
            "span",
            "",
            Number.isNaN(date.getTime())
              ? ""
              : date.toLocaleString("ru-RU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })
          )
        );
        row.appendChild(node("span", "", `${session.message_count} сообщ.`));
        if (session.has_lead) row.appendChild(node("span", "lead-badge", "заявка"));
        item.appendChild(row);
        item.addEventListener("click", () => loadSession(session.id));
        els.historyList.appendChild(item);
      });
    } catch (error) {
      els.historyList.innerHTML = "";
      els.historyList.appendChild(node("div", "history-empty", `Не удалось загрузить историю: ${error.message}`));
    }
  }

  async function loadSession(sessionId) {
    try {
      const data = await api(`/api/sessions/${sessionId}`);
      state.sessionId = data.id;
      localStorage.setItem("et_session_id", state.sessionId);
      state.leadDone = Boolean(data.lead);
      els.historyDrawer.hidden = true;
      els.body.innerHTML = "";
      els.body.appendChild(node("div", "chat-day", "История диалога"));
      (data.messages || []).forEach((message) => {
        addMessage(message.role === "user" ? "user" : "bot", message.content, message.meta || {});
      });
      if (!data.messages || !data.messages.length) renderWelcome();
      renderQuickReplies(INITIAL_QUICK);
      if (data.lead) {
        const card = leadSuccess({ lead_id: data.lead.id, summary: data.lead.summary });
        const wrapper = node("div", "msg msg-bot");
        const avatar = node("div", "msg-avatar");
        avatar.appendChild(svgIcon("i-spark"));
        wrapper.appendChild(avatar);
        const content = node("div", "msg-content");
        content.appendChild(card);
        wrapper.appendChild(content);
        els.body.appendChild(wrapper);
      }
      scrollBottom(false);
      toast("Диалог загружен.");
    } catch (error) {
      toast(error.message, true);
    }
  }

  async function clearHistoryDialog() {
    if (!confirm("Удалить всю историю сессий и заявок? Это действие необратимо.")) return;
    try {
      await api("/api/sessions/clear", { method: "POST" });
      state.sessionId = null;
      state.leadDone = false;
      localStorage.removeItem("et_session_id");
      els.historyList.innerHTML = "";
      els.body.innerHTML = "";
      newDialog();
      toast("История очищена");
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ------------------------------------------------------------------
  // Аутентификация и личный кабинет
  // ------------------------------------------------------------------

  let authMode = "login";

  function toggleAuthMode() {
    authMode = authMode === "login" ? "register" : "login";
    els.authTitle.textContent = authMode === "login" ? "Вход" : "Регистрация";
    els.authSubmit.textContent = authMode === "login" ? "Войти" : "Зарегистрироваться";
    els.authSwitch.textContent = authMode === "login" ? "Нет аккаунта? Зарегистрироваться" : "Уже есть аккаунт? Войти";
    els.authNameField.hidden = authMode === "login";
    els.authPhoneField.hidden = authMode === "login";
    els.authCompanyField.hidden = authMode === "login";
    els.authError.hidden = true;
  }

  function openAuth() {
    if (state.token && !state.user) {
      api("/api/me").then(d => {
        state.user = d.user;
        openCabinet();
      }).catch(() => {
        state.token = null;
        state.user = null;
        localStorage.removeItem("et_token");
        showAuthModal();
      });
      return;
    }
    if (state.token && state.user) {
      openCabinet();
      return;
    }
    showAuthModal();
  }

  function showAuthModal() {
    authMode = "login";
    els.authTitle.textContent = "Вход";
    els.authSubmit.textContent = "Войти";
    els.authSwitch.textContent = "Нет аккаунта? Зарегистрироваться";
    els.authNameField.hidden = true;
    els.authPhoneField.hidden = true;
    els.authCompanyField.hidden = true;
    els.authError.hidden = true;
    els.authForm.reset();
    els.authModal.hidden = false;
  }

  async function submitAuth(event) {
    event.preventDefault();
    els.authError.hidden = true;
    const form = els.authForm;
    const username = form.elements.username.value.trim();
    const password = form.elements.password.value;
    try {
      if (authMode === "register") {
        const data = await api("/api/auth/register", {
          method: "POST",
          body: JSON.stringify({
            username,
            password,
            name: form.elements.name.value.trim(),
            phone: form.elements.phone.value.trim(),
            company: form.elements.company.value.trim(),
          }),
        });
        state.token = data.token;
        state.user = data.user;
      } else {
        const data = await api("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
        state.token = data.token;
        state.user = data.user;
      }
      localStorage.setItem("et_token", state.token);
      els.authModal.hidden = true;
      updateAuthButton();
      toast("Добро пожаловать, " + (state.user.name || state.user.username));
      openCabinet();
    } catch (error) {
      els.authError.textContent = error.message;
      els.authError.hidden = false;
    }
  }

  function updateAuthButton() {
    if (state.token && state.user) {
      els.authToggle.title = state.user.name || state.user.username;
      els.authToggle.setAttribute("aria-label", "Личный кабинет");
    } else {
      els.authToggle.title = "Войти";
      els.authToggle.setAttribute("aria-label", "Войти");
    }
  }

  async function logout() {
    try { await api("/api/auth/logout", { method: "POST" }); } catch (e) { /* ignore */ }
    state.token = null;
    state.user = null;
    localStorage.removeItem("et_token");
    els.cabinetModal.hidden = true;
    updateAuthButton();
    toast("Вы вышли из аккаунта");
  }

  async function openCabinet() {
    if (!state.token) { openAuth(); return; }
    try {
      const me = await api("/api/me");
      state.user = me.user;
    } catch {
      state.token = null;
      state.user = null;
      localStorage.removeItem("et_token");
      openAuth();
      return;
    }
    const isAdmin = state.user.role === "admin";
    els.cabinetTitle.textContent = isAdmin ? "Панель администратора" : "Личный кабинет";
    els.cabinetBody.innerHTML = "";
    if (isAdmin) {
      await renderAdminCabinet();
    } else {
      await renderUserCabinet();
    }
    els.cabinetModal.hidden = false;
  }

  async function renderUserCabinet() {
    const u = state.user;
    const body = els.cabinetBody;
    body.innerHTML = "";
    const card = node("div", "lead-card");
    card.innerHTML = "<strong>Профиль</strong>";
    const p = node("div", "", "Имя: " + (u.name || "—") + " · Телефон: " + (u.phone || "—") + " · Компания: " + (u.company || "—"));
    card.appendChild(p);
    const btn = node("button", "btn-sm", "Редактировать");
    btn.onclick = () => {
      const name = prompt("Имя:", u.name || "");
      const phone = prompt("Телефон:", u.phone || "");
      const company = prompt("Компания:", u.company || "");
      if (name === null) return;
      api("/api/me", { method: "PUT", body: JSON.stringify({ name, phone, company }) })
        .then(d => { state.user = d.user; renderUserCabinet(); toast("Профиль обновлён"); })
        .catch(e => toast(e.message, true));
    };
    card.appendChild(btn);
    body.appendChild(card);

    const h3s = node("h3", "", "Мои сессии");
    body.appendChild(h3s);
    try {
      const data = await api("/api/my/sessions");
      const list = data.sessions || [];
      if (!list.length) body.appendChild(node("div", "", "Сессий пока нет."));
      list.slice(0, 10).forEach(s => {
        const row = node("div", "lead-item");
        row.innerHTML = "<strong>" + (s.title || s.id) + "</strong><br><span style='font-size:10px;color:#8c94a4'>" + s.created_at + " · " + s.message_count + " сообщ.</span>";
        body.appendChild(row);
      });
    } catch (e) {
      body.appendChild(node("div", "", "Не удалось загрузить сессии."));
    }

    const h3l = node("h3", "", "Мои заявки");
    body.appendChild(h3l);
    try {
      const data = await api("/api/my/leads");
      const leads = data.leads || [];
      if (!leads.length) body.appendChild(node("div", "", "Заявок пока нет."));
      leads.slice(0, 10).forEach(l => {
        const row = node("div", "lead-item");
        row.innerHTML = "<strong>Заявка #" + l.id + "</strong> — " + l.name + ", " + l.phone + (l.company ? ", " + l.company : "") + "<br><span style='font-size:10px;color:#8c94a4'>" + l.created_at + "</span>";
        if (l.summary) {
          const sum = node("div", "", l.summary.length > 150 ? l.summary.slice(0, 150) + "…" : l.summary);
          sum.style.fontSize = "10px";
          sum.style.color = "#8c94a4";
          sum.style.marginTop = "2px";
          row.appendChild(sum);
        }
        body.appendChild(row);
      });
    } catch (e) {
      body.appendChild(node("div", "", "Не удалось загрузить заявки."));
    }
  }

  async function renderAdminCabinet() {
    const body = els.cabinetBody;
    body.innerHTML = "";
    const h3 = node("h3", "", "Все заявки");
    body.appendChild(h3);
    try {
      const data = await api("/api/admin/leads");
      const leads = data.leads || [];
      if (!leads.length) body.appendChild(node("div", "", "Заявок пока нет."));
      leads.slice(0, 30).forEach(l => {
        const row = node("div", "lead-item");
        const title = node("div", "", "");
        title.innerHTML = "<strong>Заявка #" + l.id + "</strong> — " + l.name + ", " + l.phone + (l.company ? ", " + l.company : "");
        row.appendChild(title);
        const meta = node("div", "", l.created_at);
        meta.style.fontSize = "10px";
        meta.style.color = "#8c94a4";
        row.appendChild(meta);
        if (l.summary) {
          const sum = node("div", "", l.summary.length > 200 ? l.summary.slice(0, 200) + "…" : l.summary);
          sum.style.fontSize = "11px";
          sum.style.marginTop = "4px";
          sum.style.color = "#cfd3dc";
          row.appendChild(sum);
        }
        const btns = node("div", "", "");
        btns.style.marginTop = "6px";
        const copyBtn = node("button", "btn-sm", "Копировать контакты");
        copyBtn.onclick = () => { navigator.clipboard.writeText(l.name + " | " + l.phone + " | " + l.company); toast("Контакты скопированы"); };
        const copySumBtn = node("button", "btn-sm", "Копировать резюме");
        copySumBtn.onclick = () => { navigator.clipboard.writeText(l.summary || ""); toast("Резюме скопировано"); };
        const dlBtn = node("button", "btn-sm", "Скачать");
        dlBtn.onclick = () => {
          const blob = new Blob([JSON.stringify(l, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = "lead_" + l.id + ".json"; a.click();
          URL.revokeObjectURL(url);
        };
        const fwdBtn = node("button", "btn-sm", "Переслать менеджеру");
        fwdBtn.onclick = async () => {
          try {
            const text = "Заявка #" + l.id + "\n" + l.name + " | " + l.phone + " | " + l.company + "\n\n" + (l.summary || "");
            await navigator.clipboard.writeText(text);
            await api("/api/admin/leads/" + l.id + "/forward", { method: "POST" });
            toast("Резюме скопировано и передано");
          } catch (e) { toast(e.message, true); }
        };
        btns.appendChild(copyBtn);
        btns.appendChild(copySumBtn);
        btns.appendChild(dlBtn);
        btns.appendChild(fwdBtn);
        row.appendChild(btns);
        body.appendChild(row);
      });
    } catch (e) {
      body.appendChild(node("div", "", "Не удалось загрузить заявки: " + e.message));
    }

    const h3s = node("h3", "", "Все сессии");
    body.appendChild(h3s);
    try {
      const data = await api("/api/admin/sessions");
      const sessions = data.sessions || [];
      if (!sessions.length) body.appendChild(node("div", "", "Сессий пока нет."));
      sessions.slice(0, 20).forEach(s => {
        const row = node("div", "lead-item");
        row.innerHTML = "<strong>" + (s.title || s.id) + "</strong><br><span style='font-size:10px;color:#8c94a4'>" + s.created_at + " · " + s.message_count + " сообщ. · " + (s.has_lead ? "заявка" : "без заявки") + "</span>";
        body.appendChild(row);
      });
    } catch (e) {
      body.appendChild(node("div", "", "Не удалось загрузить сессии."));
    }
  }
  async function newDialog() {
    try {
      const data = await api("/api/session/new", { method: "POST" });
      state.sessionId = data.session_id;
      localStorage.setItem("et_session_id", state.sessionId);
      state.leadDone = false;
      state.lastDebug = null;
      renderWelcome();
      els.historyDrawer.hidden = true;
      if (state.debug) updateDebugPanel(null);
      toast("Новый диалог начат — контекст сброшен.");
      els.input.focus();
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ------------------------------------------------------------------
  // Инициализация
  // ------------------------------------------------------------------

  function resizeInput() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 108) + "px";
  }

  function focusChat() {
    const chat = $("#chat");
    if (!chat) return;
    chat.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => els.input.focus({ preventScroll: true }), 500);
  }

  async function init() {
    els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(els.input.value);
    });

    els.input.addEventListener("input", resizeInput);
    els.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(els.input.value);
      }
    });

    els.debugToggle.addEventListener("click", toggleDebug);
    els.historyToggle.addEventListener("click", () => {
      if (els.historyDrawer.hidden) openHistory();
      else els.historyDrawer.hidden = true;
    });
    els.historyClose.addEventListener("click", () => {
      els.historyDrawer.hidden = true;
    });
    if (els.clearHistory) {
      els.clearHistory.addEventListener("click", clearHistoryDialog);
    }
    els.newDialog.addEventListener("click", newDialog);
    els.authToggle.addEventListener("click", openAuth);
    els.authClose.addEventListener("click", () => { els.authModal.hidden = true; });
    els.authSwitch.addEventListener("click", toggleAuthMode);
    els.authForm.addEventListener("submit", submitAuth);
    els.cabinetClose.addEventListener("click", () => { els.cabinetModal.hidden = true; });
    els.cabinetLogout.addEventListener("click", logout);
    els.cabinetLogout.addEventListener("click", logout);

    $$("[data-focus-chat]").forEach((element) => {
      element.addEventListener("click", (event) => {
        event.preventDefault();
        focusChat();
      });
    });

    $$(".product-card[data-topic]").forEach((card) => {
      card.addEventListener("click", (event) => {
        if (event.target.closest("a, button")) return;
        els.input.value = `Расскажите подробнее: ${card.dataset.topic}`;
        resizeInput();
        focusChat();
      });
    });

    // Восстанавливаем последний диалог, если он есть.
    if (state.sessionId) {
      try {
        const data = await api(`/api/sessions/${state.sessionId}`);
        if (data.messages && data.messages.length) {
          els.body.innerHTML = "";
          els.body.appendChild(node("div", "chat-day", "История диалога"));
          data.messages.forEach((message) => {
            addMessage(message.role === "user" ? "user" : "bot", message.content, message.meta || {});
          });
          state.leadDone = Boolean(data.lead);
          renderQuickReplies(INITIAL_QUICK);
          scrollBottom(false);
        } else {
          renderWelcome();
        }
      } catch {
        state.sessionId = null;
        localStorage.removeItem("et_session_id");
        renderWelcome();
      }
    } else {
      renderWelcome();
    }

    // Проверяем режим работы и обновляем подпись.
    try {
      const health = await api("/api/health");
      if (health.llm_enabled) {
        els.caption.innerHTML = "";
        const dot = node("span", "live-dot");
        els.caption.appendChild(dot);
        els.caption.appendChild(
          node("span", "", `LLM подключена (${health.llm_model}) · ответы по базе знаний с источниками`)
        );
      }
    } catch {
      // Сервер может быть ещё не готов — не мешаем работе чата.
    }

    // Плавное появление блоков при скролле.
    if ("IntersectionObserver" in window) {
      const targets = $$(".section-head, .product-card, .step, .why-copy, .why-item, .cta-inner");
      targets.forEach((element) => element.classList.add("reveal"));
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-visible");
              observer.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.12 }
      );
      targets.forEach((element) => observer.observe(element));
    }

    resizeInput();
  }

  document.addEventListener("DOMContentLoaded", init);
  window.clearHistoryDialog = clearHistoryDialog;
  window.logout = logout;
  window.openAuth = openAuth;
  window.toggleAuthMode = toggleAuthMode;
  window.submitAuth = submitAuth;
  window.openCabinet = openCabinet;
})();
