async function loadStats() {
  const s = await api("/api/stats");
  renderStats($("stats"), s);
  const tickets = el("div", "stat");
  tickets.append(el("strong", null, s.open_tickets), el("span", null, "Open tickets"));
  const escalated = el("div", "stat escalated");
  escalated.append(el("strong", null, s.escalations), el("span", null, "Frustrated customers escalated"));
  $("stats").append(tickets, escalated);
}

async function loadGaps() {
  const gaps = await api("/api/gaps");
  const box = $("gaps");
  box.replaceChildren();
  if (!gaps.length) {
    box.append(el("p", "muted", "No open gaps. The bot answered everything on-topic 🎉"));
    return;
  }
  for (const gap of gaps) {
    const card = el("form", "gap");
    const head = el("div", "gap-head");
    const title = el("strong", null, gap.label);
    if (gap.partial) title.append(el("span", "tag", `partly answered ${gap.partial}×`));
    head.append(title, el("span", "count", `asked ${gap.count}×`));
    card.append(head);

    const qs = el("ul", "gap-questions");
    for (const q of gap.questions) qs.append(el("li", null, q));
    card.append(qs);

    const intent = el("input");
    intent.value = gap.suggested_intent;
    intent.required = true;
    intent.title = "Intent name";
    const answer = el("textarea");
    answer.placeholder = "Write the answer the bot should give…";
    answer.required = true;
    answer.rows = 2;
    const status = el("span", "muted");
    const save = el("button", null, "Teach the bot");
    save.type = "submit";

    const row = el("div", "gap-row");
    row.append(el("label", "muted", "Intent"), intent);
    card.append(row, answer, save, status);

    card.addEventListener("submit", async (e) => {
      e.preventDefault();
      save.disabled = true;
      try {
        const r = await api("/api/gaps/resolve", {
          method: "POST",
          body: JSON.stringify({ intent: intent.value.trim(), patterns: gap.questions, answer: answer.value.trim() }),
        });
        status.textContent = ` Saved as "${r.saved}" with ${r.patterns} example phrasings.`;
        setTimeout(refresh, 1200);
      } catch (err) {
        status.textContent = ` Error: ${err.message}`;
        save.disabled = false;
      }
    });
    box.append(card);
  }
}

async function loadTickets() {
  const tickets = await api("/api/tickets");
  const list = $("tickets");
  list.replaceChildren();
  for (const t of tickets) {
    const li = el("li", t.status === "closed" ? "muted" : null);
    const text = el("span");
    text.append(el("strong", null, `#${t.id} `), `${t.question} `, el("span", "muted", `· ${t.email}`));
    li.append(text);
    if (t.status === "open") {
      const close = el("button", "small", "Close");
      close.onclick = async () => { await api(`/api/tickets/${t.id}/close`, { method: "POST" }); refresh(); };
      li.append(close);
    } else {
      li.append(el("span", "muted", "closed"));
    }
    list.append(li);
  }
  if (!tickets.length) list.append(el("li", "muted", "No tickets yet"));
}

async function loadLogs() {
  const logs = await api("/api/logs?limit=30");
  const body = $("logs");
  body.replaceChildren();
  for (const l of logs) {
    const tr = el("tr");
    const badge = el("span", `badge ${l.route}`, ROUTE_LABEL[l.route] || l.route);
    const routeCell = el("td");
    routeCell.append(badge);
    tr.append(el("td", "muted", l.created_at.slice(5, 16)), el("td", null, l.user_message), routeCell,
      el("td", null, l.confidence?.toFixed(2) ?? ""), el("td", "muted", l.bot_reply.slice(0, 90)));
    body.append(tr);
  }
}

async function loadModel() {
  const m = await api("/api/model");
  const box = $("model");
  box.replaceChildren();
  const pct = (x) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`);
  const tiles = [
    [m.examples?.toLocaleString(), "training examples"],
    [m.intents, "intents"],
    [pct(m.accuracy), "accuracy, dataset test split"],
    [pct(m.real_world_accuracy), `accuracy, ${m.real_world_size ?? 0} hand-written questions`],
  ];
  for (const [value, label] of tiles) {
    const stat = el("div", "stat");
    stat.append(el("strong", null, value ?? "-"), el("span", null, label));
    box.append(stat);
  }
}

async function loadEvaluation() {
  const e = await api("/api/evaluation");
  const box = $("evaluation");
  box.replaceChildren();
  if (!e.total) {
    box.append(el("p", "muted", "No evaluation yet. Run: python -m app.evaluate"));
    return;
  }
  box.append(el("p", null, `${e.correct} of ${e.total} test questions handled correctly (${(e.accuracy * 100).toFixed(1)}%). Run on ${e.run_at}.`));
  if (e.holdout) {
    box.append(el("p", "muted", `Held-out set (questions never used while tuning the bot): ${e.holdout.correct} of ` +
      `${e.holdout.total} (${(e.holdout.accuracy * 100).toFixed(1)}%). This is the realistic estimate for new customers.`));
  }
  const table = el("table", "eval-table");
  const head = el("tr");
  for (const h of ["Expected route", "Questions", "Correct", "Accuracy"]) head.append(el("th", null, h));
  table.append(head);
  for (const [route, r] of Object.entries(e.by_route)) {
    const tr = el("tr");
    tr.append(el("td", null, ROUTE_LABEL[route] || route), el("td", null, r.total), el("td", null, r.correct),
      el("td", null, `${(r.accuracy * 100).toFixed(0)}%`));
    table.append(tr);
  }
  box.append(table);
}

function refresh() {
  loadModel();
  loadEvaluation();
  loadStats();
  loadGaps();
  loadTickets();
  loadLogs();
}

setupUpload();
loadDocs();
refresh();
