let sessionId = null;
try { sessionId = sessionStorage.getItem("session_id"); } catch {}

function scroll() { $("messages").scrollTop = $("messages").scrollHeight; }

function addUser(text) {
  $("messages").append(el("div", "msg user", text));
  scroll();
}

function ticketForm(question) {
  const form = el("form", "ticket");
  const email = el("input");
  email.type = "email";
  email.placeholder = "you@example.com";
  email.required = true;
  const button = el("button", "small", "Create ticket");
  button.type = "submit";
  form.append(email, button);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    button.disabled = true;
    try {
      const t = await api("/api/tickets", {
        method: "POST",
        body: JSON.stringify({ question, email: email.value, session_id: sessionId }),
      });
      form.replaceWith(el("div", "muted note", `✓ Ticket #${t.id} created. Our team will email you.`));
    } catch (err) {
      button.disabled = false;
      button.textContent = "Retry";
      email.setCustomValidity(err.message);
      email.reportValidity();
      email.setCustomValidity("");
    }
  });
  return form;
}

function addBot(data, question) {
  const msg = el("div", "msg bot");
  if (data.corrected) msg.append(el("div", "muted note", `Showing results for "${data.corrected}"`));

  const body = el("div");
  // Highlight the exact span the QA model extracted inside the cited sentence.
  const i = data.highlight ? data.reply.indexOf(data.highlight) : -1;
  if (i >= 0) {
    body.append(data.reply.slice(0, i), el("mark", null, data.highlight), data.reply.slice(i + data.highlight.length));
  } else {
    body.textContent = data.reply;
  }
  msg.append(body);

  if (data.products?.length) {
    const cards = el("div", "products");
    for (const p of data.products) {
      const card = el("div", "product");
      const stock = p.stock === 0 ? "Out of stock" : p.stock < 10 ? `Only ${p.stock} left` : "In stock";
      card.append(
        el("strong", null, p.name),
        el("div", "price", `₹${p.price.toLocaleString("en-IN")}`),
        el("div", `stock${p.stock === 0 ? " out" : ""}`, stock),
        el("div", "muted desc", p.description),
      );
      const ask = el("button", "chip", "Ask about this");
      ask.type = "button";
      ask.onclick = () => send(`tell me about the ${p.name.toLowerCase()}`);
      card.append(ask);
      cards.append(card);
    }
    msg.append(cards);
  }

  if (data.suggestions?.length) {
    const chips = el("div", "chips");
    for (const s of data.suggestions) {
      const chip = el("button", "chip", s.label);
      chip.type = "button";
      chip.onclick = () => send(s.query);
      chips.append(chip);
    }
    msg.append(chips);
  }

  if (data.offer_ticket) msg.append(ticketForm(question));

  const meta = el("div", "meta");
  meta.append(el("span", `badge ${data.route}`, ROUTE_LABEL[data.route]));
  if (data.escalated) meta.append(el("span", "badge escalated", "Escalated to human"));
  if (data.intent && data.route === "faq") meta.append(el("span", null, `intent: ${data.intent}`));
  if (data.confidence) meta.append(el("span", null, `confidence ${data.confidence}`));
  msg.append(meta);

  if (data.sources?.length) {
    const details = el("details", "sources");
    details.append(el("summary", null, `${data.sources.length} source${data.sources.length > 1 ? "s" : ""}`));
    for (const s of data.sources) {
      const src = el("div", `source${s.used ? " used" : ""}`);
      src.append(
        el("b", null, `📄 ${s.document}${s.section ? " › " + s.section : ""} · p.${s.page}`),
        el("span", "muted", `  similarity ${s.similarity}${s.used ? " · cited" : ""}`),
        el("div", "muted", s.snippet),
      );
      details.append(src);
    }
    msg.append(details);
  }
  $("messages").append(msg);
  scroll();
}

async function send(text) {
  addUser(text);
  const button = $("chat-form").querySelector("button");
  button.disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    sessionId = data.session_id;
    try { sessionStorage.setItem("session_id", sessionId); } catch {}
    addBot(data, text);
    loadStats();
    if (data.route === "action") loadOrders();
  } catch (e) {
    addBot({ reply: `Error: ${e.message}`, route: "fallback", confidence: 0 });
  } finally {
    button.disabled = false;
  }
}

async function loadStats() {
  renderStats($("stats"), await api("/api/stats"));
}

async function loadOrders() {
  const orders = await api("/api/orders?limit=40");
  const list = $("orders");
  list.replaceChildren();
  for (const o of orders) {
    const li = el("li");
    li.append(el("span", null, `#${o.id} · ${o.customer} · ₹${o.total.toLocaleString("en-IN")}`),
      el("span", "status", o.status.replaceAll("_", " ")));
    li.onclick = () => send(`where is order ${o.id}`);
    list.append(li);
  }
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  send(text);
});

document.querySelectorAll(".suggestions li").forEach((li) => li.addEventListener("click", () => send(li.textContent)));

addBot({
  reply: "Hi! I'm the ShopEase assistant. I can track, cancel or refund your orders, send invoices, and answer questions about shipping, payments and returns, showing you the source.",
  route: "faq",
});
loadStats();
loadOrders();
