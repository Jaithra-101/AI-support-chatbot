// Helpers shared by the chat page and the admin page.
const $ = (id) => document.getElementById(id);

const ROUTE_LABEL = {
  action: "Order system",
  product: "Product catalog",
  intent: "Trained intent",
  faq: "FAQ",
  rag: "Documents",
  passage: "Closest passage",
  clarify: "Clarifying",
  offtopic: "Off-topic",
  fallback: "Knowledge gap",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(", ") : data.detail;
    throw new Error(detail || res.statusText);
  }
  return data;
}

function renderStats(box, stats) {
  box.replaceChildren();
  for (const route of Object.keys(ROUTE_LABEL)) {
    const stat = el("div", `stat ${route}`);
    stat.append(el("strong", null, stats.by_route[route] || 0), el("span", null, ROUTE_LABEL[route]));
    box.append(stat);
  }
}

async function loadDocs(onChange) {
  const docs = await api("/api/documents");
  const list = $("docs");
  list.replaceChildren();
  for (const d of docs) {
    const li = el("li");
    li.append(el("span", null, `${d.name} (${d.chunks} chunks)`));
    const del = el("button", "del", "✕");
    del.title = "Remove document";
    del.onclick = async () => { await api(`/api/documents/${d.id}`, { method: "DELETE" }); loadDocs(onChange); onChange?.(); };
    li.append(del);
    list.append(li);
  }
  if (!docs.length) list.append(el("li", "muted", "No documents yet"));
}

function setupUpload(onChange) {
  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    $("upload-status").textContent = "Indexing…";
    try {
      const data = await api("/api/documents", { method: "POST", body: form });
      $("upload-status").textContent = `Added ${data.name}: ${data.chunks} chunks`;
    } catch (err) {
      $("upload-status").textContent = `Error: ${err.message}`;
    }
    $("file").value = "";
    loadDocs(onChange);
    onChange?.();
  });
}
