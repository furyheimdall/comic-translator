// 만화 번역기 web UI (no build step).
const view = document.getElementById("view");
const state = { meta: null, providers: [], pollTimer: null, routeVersion: 0 };

const STATUS = { queued: "대기", running: "실행 중", done: "완료", failed: "실패", cancelled: "취소됨" };

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, options = {}) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (options.json !== undefined) {
    init.body = JSON.stringify(options.json);
    init.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, init);
  if (res.status === 401 && !path.startsWith("/api/login")) {
    renderLogin();
    throw new Error("로그인이 필요합니다.");
  }
  const type = res.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail || body || `HTTP ${res.status}`);
  return body;
}

function stopPolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function toast(message, isError = false) {
  const el = h("div", { class: `toast ${isError ? "error" : ""}` }, message);
  document.body.append(el);
  setTimeout(() => el.remove(), 5000);
}

function fmtTime(ts) {
  return ts ? new Date(ts * 1000).toLocaleString("ko-KR") : "";
}

// ------------------------------------------------------------------ routing
async function route() {
  stopPolling();
  const version = ++state.routeVersion;
  document.onkeydown = null;
  view.replaceChildren(h("p", { class: "muted" }, "불러오는 중…"));
  const hash = location.hash || "#/jobs";
  const [, section, id] = hash.split("/");
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === section));
  try {
    const session = await api("/api/session");
    if (version !== state.routeVersion) return;
    if (!session.authenticated) return renderLogin();
    if (!state.meta) state.meta = await api("/api/meta");
    state.providers = await api("/api/providers");
    if (version !== state.routeVersion) return;
    if (section === "new") return await renderNewJob(version);
    if (section === "models") return await renderModelSetup(version, id ? decodeURIComponent(id) : null);
    if (section === "providers") return renderProviders();
    if (section === "job" && id) return await renderJob(id, version);
    return await renderJobs(version);
  } catch (err) {
    if (version !== state.routeVersion) return;
    view.replaceChildren(h("p", { class: "error" }, err.message));
  }
}

function renderLogin() {
  stopPolling();
  ++state.routeVersion;
  view.replaceChildren(document.getElementById("login-tpl").content.cloneNode(true));
  document.getElementById("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const password = new FormData(event.target).get("password");
    try {
      await api("/api/login", { method: "POST", json: { password } });
      route();
    } catch (err) {
      document.getElementById("login-error").textContent = err.message;
    }
  });
}

// ------------------------------------------------------------------ jobs list
async function renderJobs(version) {
  const jobs = await api("/api/jobs");
  if (version !== state.routeVersion) return;
  const providerName = (id) => state.providers.find((p) => p.id === id)?.name || "(삭제됨)";
  const engineName = (id) => state.meta.engines.find((e) => e.id === id)?.name || id;
  const rows = jobs.map((job) =>
    h(
      "tr",
      { class: "clickable", onclick: () => (location.hash = `#/job/${job.id}`) },
      h("td", {}, job.title),
      h("td", {}, h("span", { class: `badge ${job.status}` }, STATUS[job.status])),
      h("td", {}, `${job.translated_count}/${job.page_count}`),
      h("td", {}, engineName(job.engine)),
      h("td", {}, providerName(job.provider_id)),
      h("td", {}, fmtTime(job.created_at)),
    ),
  );
  view.replaceChildren(
    h(
      "section",
      { class: "card" },
      h("div", { class: "row between" }, h("h2", {}, "작업"), h("a", { class: "button", href: "#/new" }, "새 번역")),
      jobs.length
        ? h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["제목", "상태", "페이지", "엔진", "LLM", "생성"].map((t) => h("th", {}, t)))),
            h("tbody", {}, rows),
          )
        : h("p", { class: "muted" }, "아직 작업이 없습니다."),
    ),
  );
  const refresh = async () => {
    try {
      const latest = await api("/api/jobs");
      if (version !== state.routeVersion) return;
      if (latest.length !== jobs.length || latest.some((job, i) => job.id !== jobs[i].id)) {
        return route();
      }
      latest.forEach((job, i) => {
        const badge = rows[i].querySelector(".badge");
        badge.className = `badge ${job.status}`;
        badge.textContent = STATUS[job.status];
        rows[i].children[2].textContent = `${job.translated_count}/${job.page_count}`;
      });
      if (latest.some((job) => ["running", "queued"].includes(job.status))) {
        state.pollTimer = setTimeout(refresh, 3000);
      }
    } catch (err) {
      if (version === state.routeVersion) toast(err.message, true);
    }
  };
  if (jobs.some((job) => ["running", "queued"].includes(job.status))) {
    state.pollTimer = setTimeout(refresh, 3000);
  }
}

// ------------------------------------------------------------------ new job
function optionField(engine, option) {
  const name = `opt:${engine.id}:${option.key}`;
  let input;
  if (option.type === "select") {
    input = h(
      "select",
      { name },
      option.choices.map((c) => h("option", { value: c.value, selected: c.value === option.default }, c.label)),
    );
  } else if (option.type === "bool") {
    input = h("input", { type: "checkbox", name, checked: option.default });
  } else {
    input = h("input", { type: "number", name, value: option.default, min: option.min ?? undefined, max: option.max ?? undefined });
  }
  return h(
    "label",
    { class: option.type === "bool" ? "check-field" : "" },
    option.type === "bool" ? h("span", { class: "check" }, input, " ", option.label) : [option.label, input],
    option.help ? h("small", { class: "muted" }, option.help) : null,
  );
}

async function renderNewJob(version) {
  const connected = state.providers.filter((p) => p.connected);
  if (!connected.length) {
    view.replaceChildren(
      h(
        "section",
        { class: "card" },
        h("h2", {}, "새 번역"),
        h("p", {}, "먼저 사용할 LLM 제공자를 등록하고 연결하세요. "),
        h("a", { class: "button", href: "#/providers" }, "LLM 제공자 설정"),
      ),
    );
    return;
  }
  // Model readiness is advisory only: an engine may already run from models
  // cached elsewhere, so a missing record never blocks job creation.
  const modelSetup = await api("/api/model-setup").catch(() => null);
  if (version !== state.routeVersion) return;
  const engines = state.meta.engines;
  const firstAvailable = engines.find((e) => e.available) || engines[0];
  const engineOptions = h("div", { id: "engine-options" });

  const showEngineOptions = (engineId) => {
    const engine = engines.find((e) => e.id === engineId);
    engineOptions.replaceChildren(
      ...[
        h("p", { class: "muted" }, engine.description),
        engine.available ? null : h("p", { class: "error" }, `사용 불가: ${engine.reason}`),
        modelHint(engine, modelSetup),
        h("div", { class: "grid" }, engine.options.map((o) => optionField(engine, o))),
      ].filter(Boolean),
    );
  };

  const fileInput = h("input", { type: "file", name: "files", multiple: true, accept: ".png,.jpg,.jpeg,.webp,.zip,.cbz,.pdf", required: true });
  const fileList = h("p", { class: "muted" }, "이미지 여러 장, ZIP/CBZ, PDF 파일을 선택하세요. 파일 이름 순서대로 처리합니다.");
  fileInput.addEventListener("change", () => {
    const files = [...fileInput.files];
    const size = files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
    fileList.textContent = `${files.length}개 파일, ${size.toFixed(1)} MB`;
  });

  const progress = h("progress", { max: 100, value: 0, hidden: true });
  const form = h(
    "form",
    { class: "stack" },
    h("label", {}, "제목", h("input", { name: "title", placeholder: "예: 작품명 3권" })),
    h("label", {}, "원고", fileInput, fileList),
    h(
      "div",
      { class: "grid" },
      h(
        "label",
        {},
        "이미지 처리 엔진",
        h(
          "select",
          { name: "engine", onchange: (e) => showEngineOptions(e.target.value) },
          engines.map((e) => h("option", { value: e.id, selected: e.id === firstAvailable.id }, `${e.name}${e.available ? "" : " (설치 필요)"}`)),
        ),
      ),
      h(
        "label",
        {},
        "LLM 제공자",
        h("select", { name: "provider_id" }, connected.map((p) => h("option", { value: p.id }, `${p.name} · ${p.model}`))),
      ),
    ),
    engineOptions,
    h(
      "label",
      {},
      "작품 용어집·인물 메모 (선택)",
      h("textarea", {
        name: "instructions",
        rows: 5,
        placeholder: "美咲 → 미사키 (여, 주인공의 선배, 츤데레)\n翔太 → 쇼타 (남, 미사키에게 존댓말)\n生徒会 → 학생회",
      }),
      h("small", { class: "muted" }, "기본 한국어 번역 지침에 덧붙여 모든 페이지 번역에 전달됩니다."),
    ),
    h("button", { type: "submit" }, "번역 시작"),
    progress,
  );
  showEngineOptions(firstAvailable.id);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const engineId = data.get("engine");
    const engine = engines.find((e) => e.id === engineId);
    const options = {};
    for (const option of engine.options) {
      const field = form.elements[`opt:${engineId}:${option.key}`];
      options[option.key] = option.type === "bool" ? field.checked : option.type === "int" ? Number(field.value) : field.value;
    }
    const body = new FormData();
    for (const file of fileInput.files) body.append("files", file, file.name);
    body.append("engine", engineId);
    body.append("provider_id", data.get("provider_id"));
    body.append("title", data.get("title"));
    body.append("instructions", data.get("instructions"));
    body.append("options", JSON.stringify(options));

    // XHR for upload progress.
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    progress.hidden = false;
    form.querySelector("button[type=submit]").disabled = true;
    xhr.upload.onprogress = (e) => e.lengthComputable && (progress.value = (e.loaded / e.total) * 100);
    xhr.onload = () => {
      const res = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 400) {
        toast(res.detail || `업로드 실패 (HTTP ${xhr.status})`, true);
        form.querySelector("button[type=submit]").disabled = false;
        progress.hidden = true;
        return;
      }
      location.hash = `#/job/${res.id}`;
    };
    xhr.onerror = () => toast("업로드 실패", true);
    xhr.send(body);
  });

  view.replaceChildren(h("section", { class: "card" }, h("h2", {}, "새 번역"), form));
}

// ------------------------------------------------------------------ job detail
async function downloadZip(job, button) {
  button.disabled = true;
  button.textContent = "ZIP 준비 중…";
  let url;
  try {
    // CBZ is a ZIP archive; reuse the server archive without recompressing.
    const response = await fetch(`/api/jobs/${job.id}/download`);
    if (!response.ok) throw new Error(`다운로드 실패 (HTTP ${response.status})`);
    url = URL.createObjectURL(await response.blob());
    const link = h("a", { href: url, download: `${job.title.replace(/[\\/:*?"<>|]/g, "_")} (한국어).zip` });
    document.body.append(link);
    link.click();
    link.remove();
  } catch (err) {
    toast(err.message, true);
  } finally {
    if (url) setTimeout(() => URL.revokeObjectURL(url), 60000);
    button.disabled = false;
    button.textContent = "ZIP 다운로드";
  }
}

async function renderJob(jobId, version) {
  let job = await api(`/api/jobs/${jobId}`);
  if (version !== state.routeVersion) return;
  const engine = state.meta.engines.find((e) => e.id === job.engine);
  const providerLabel = h("span");
  const badge = h("span");
  const count = h("span");
  const progress = h("progress", { max: 1 });
  const message = h("p", { class: "muted" });
  const error = h("pre", { class: "error" });
  const missingPages = h("p", { class: "error" });
  const connected = state.providers.filter((provider) => provider.connected);
  const providerSelect = h("select", { name: "retry_provider_id" },
    h("option", { value: "", selected: !connected.some((provider) => provider.id === job.provider_id) }, "LLM 제공자를 선택하세요"),
    connected.map((provider) => h("option", { value: provider.id, selected: provider.id === job.provider_id }, `${provider.name} · ${provider.model}`)));
  const retrySettings = h("div", { class: "stack" },
    h("label", {}, "재시도에 사용할 LLM", providerSelect),
    h("p", { class: "muted" }, "선택한 제공자의 모델·추가 요청 설정을 사용합니다. 부분 재시도는 완료된 이미지를 유지하므로 페이지별 번역 스타일이 달라질 수 있습니다. ",
      h("a", { href: "#/providers" }, "모델·API 설정 수정")));
  providerSelect.addEventListener("change", () => {
    retry.disabled = retryFailed.disabled = !providerSelect.value;
  });
  const refreshAfter = async (action) => {
    const json = action === "cancel" ? undefined : { provider_id: providerSelect.value };
    if (json && !json.provider_id) throw new Error("재시도에 사용할 LLM 제공자를 선택하세요.");
    await api(`/api/jobs/${jobId}/${action}`, { method: "POST", json });
    if (version !== state.routeVersion) return;
    stopPolling();
    await refresh();
  };
  const cancel = h("button", { class: "secondary", onclick: () => refreshAfter("cancel").catch((err) => toast(err.message, true)) }, "취소");
  const retry = h("button", { class: "secondary", onclick: () => {
    if (confirm(`기존 번역 결과를 모두 지우고 ${providerSelect.selectedOptions[0].textContent}(으)로 다시 실행할까요?`)) refreshAfter("retry").catch((err) => toast(err.message, true));
  } }, "전체 다시 실행");
  const retryFailed = h("button", { onclick: async () => {
    retryFailed.disabled = true;
    try { await refreshAfter("retry-failed"); }
    catch (err) { toast(err.message, true); }
    finally { retryFailed.disabled = !providerSelect.value; }
  } }, "실패·미완료 페이지만 재시도");
  const cbz = h("a", { class: "button", href: `/api/jobs/${jobId}/download` }, "CBZ 다운로드");
  const zip = h("button", { onclick: (event) => downloadZip(job, event.currentTarget) }, "ZIP 다운로드");
  const remove = h("button", { class: "danger", onclick: async () => {
    if (confirm("작업과 파일을 삭제할까요?")) {
      await api(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (version === state.routeVersion) location.hash = "#/jobs";
    }
  } }, "삭제");
  const log = h("pre", { class: "log" }, "불러오는 중…");
  const details = h("details", {}, h("summary", {}, "작업 로그"), log);
  details.addEventListener("toggle", async () => {
    if (details.open) {
      try { log.textContent = (await api(`/api/jobs/${jobId}/log`)) || "(비어 있음)"; }
      catch (err) { log.textContent = err.message; }
    }
  });
  const viewer = pageViewer(job);
  const header = h("section", { class: "card" },
    h("div", { class: "row between" }, h("h2", {}, job.title),
      h("div", { class: "row" }, cancel, retryFailed, retry, zip, cbz, remove)),
    h("p", {}, badge, ` ${engine?.name || job.engine} · `, providerLabel, " · ", count),
    progress, message, error, missingPages, retrySettings, details);
  const update = () => {
    const active = ["running", "queued"].includes(job.status);
    badge.className = `badge ${job.status}`;
    badge.textContent = STATUS[job.status];
    const provider = state.providers.find((item) => item.id === job.provider_id);
    providerLabel.textContent = provider ? `${provider.name} (${provider.model})` : "제공자 삭제됨";
    retrySettings.hidden = active;
    retry.disabled = retryFailed.disabled = !providerSelect.value;
    count.textContent = `${job.translated_count}/${job.page_count}쪽`;
    progress.value = job.progress;
    progress.hidden = !active;
    message.textContent = job.message || "";
    message.hidden = !job.message;
    error.textContent = job.error || "";
    error.hidden = !job.error;
    cancel.hidden = !active;
    retry.hidden = remove.hidden = active;
    const missing = job.pages.filter((page) => !page.translated);
    retryFailed.hidden = active || !missing.length;
    retryFailed.textContent = `실패·미완료 ${missing.length}쪽만 재시도`;
    missingPages.hidden = active || !missing.length;
    missingPages.textContent = `실패·미완료 페이지: ${missing.map((page) => page.idx).join(", ")}쪽. 재시도 시 완료된 이미지는 보존됩니다.`;
    zip.hidden = cbz.hidden = !job.page_count || job.translated_count !== job.page_count;
    viewer.update(job);
    return active;
  };
  const refresh = async () => {
    try {
      const latest = await api(`/api/jobs/${jobId}`);
      if (version !== state.routeVersion) return;
      job = latest;
      if (update()) state.pollTimer = setTimeout(refresh, 3000);
    } catch (err) {
      if (version === state.routeVersion) toast(err.message, true);
    }
  };
  view.replaceChildren(header, viewer.element);
  if (update()) state.pollTimer = setTimeout(refresh, 3000);
}

function pageViewer(initialJob) {
  let job = initialJob;
  let idx = 1;
  let showOriginal = false;
  const img = h("img", { class: "page" });
  const label = h("span");
  const toggle = h("button", { class: "secondary", onclick: () => { showOriginal = !showOriginal; update(); } });
  const next = h("button", { class: "secondary", onclick: () => go(1) }, "← 다음");
  const previous = h("button", { class: "secondary", onclick: () => go(-1) }, "이전 →");
  const update = () => {
    const page = job.pages.find((p) => p.idx === idx);
    img.hidden = !page;
    toggle.disabled = !page?.translated;
    next.disabled = idx >= job.page_count;
    previous.disabled = idx <= 1;
    if (!page) { label.textContent = "페이지가 없습니다."; return; }
    const original = showOriginal || !page.translated;
    const src = `/api/jobs/${job.id}/pages/${idx}/${original ? "source" : "output"}?v=${job.started_at || job.created_at}`;
    if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    img.alt = page.source_name;
    toggle.textContent = original ? "번역본 보기" : "원본 보기";
    label.textContent = `${idx} / ${job.page_count} · ${page.source_name}${page.translated ? "" : " (미번역)"}`;
  };
  const go = (delta) => {
    const target = Math.max(1, Math.min(job.page_count, idx + delta));
    if (target !== idx) { idx = target; update(); }
  };
  img.addEventListener("click", (event) => {
    const rect = img.getBoundingClientRect();
    go(event.clientX - rect.left < rect.width / 2 ? 1 : -1);
  });
  document.onkeydown = (event) => {
    if (location.hash !== `#/job/${job.id}` || event.target.closest("input, textarea, select, button, a, summary, [contenteditable]")) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); go(1); }
    if (event.key === "ArrowRight") { event.preventDefault(); go(-1); }
    if (event.key === " " && !toggle.disabled) { event.preventDefault(); showOriginal = !showOriginal; update(); }
  };
  return {
    element: h("section", { class: "card viewer" }, h("div", { class: "row between" }, next, label, toggle, previous), img,
      h("p", { class: "muted" }, "이미지 왼쪽 클릭/← : 다음 쪽, 오른쪽 클릭/→ : 이전 쪽, Space: 원본·번역 전환")),
    update(latest) { job = latest; update(); },
  };
}

// ------------------------------------------------------------------ model setup
// Engine installation (setup scripts) and model files (downloaded here, only on
// explicit request) are separate steps; the UI keeps both states visible.
const PREP_STATE = { idle: "다운로드 대기", running: "다운로드 진행 중", done: "다운로드 완료", failed: "다운로드 실패" };
const PREP_BADGE = { idle: "queued", running: "running", done: "done", failed: "failed" };

function modelHint(engine, setup) {
  const link = h("a", { href: `#/models/${encodeURIComponent(engine.id)}` }, "모델 준비 화면");
  if (!engine.available) return h("p", {}, "엔진 설치 상태와 필요한 모델은 ", link, "에서 확인할 수 있습니다.");
  const defaults = (setup?.profiles || []).filter((p) => p.engine === engine.id && p.default);
  if (!defaults.length || defaults.every((p) => p.prepared)) return null;
  return h(
    "p",
    { class: "muted" },
    "이 엔진의 기본 모델 파일을 설정한 저장 위치에서 확인하지 못했습니다. 새 설치의 번역 작업을 시작하려면 ",
    link,
    "에서 먼저 필요한 모델을 준비하세요. 기존 설치의 작업은 이전 캐시를 그대로 사용할 수 있습니다.",
  );
}

function hfModelUrl(model) {
  if (typeof model.url === "string" && model.url.startsWith("https://")) return model.url;
  return `https://huggingface.co/${model.repo.split("/").map(encodeURIComponent).join("/")}`;
}

function fileSummary(files, limit = 3) {
  if (!files?.length) return "저장소 전체";
  const shown = files.slice(0, limit).join(", ");
  return files.length > limit ? `${shown} 외 ${files.length - limit}개` : shown;
}

async function renderModelSetup(version, focusEngine) {
  let setup = await api("/api/model-setup");
  if (version !== state.routeVersion) return;
  const engineMeta = (id) => state.meta.engines.find((e) => e.id === id);
  const isRunning = () => setup.status?.state === "running";

  // --- storage + token form
  const cacheInput = h("input", { id: "model-cache-dir", name: "cache_dir", class: "mono", value: setup.cache_dir || "", required: true, autocomplete: "off", spellcheck: "false", "aria-describedby": "model-cache-help" });
  const tokenInput = h("input", { id: "model-hf-token", name: "token", type: "password", autocomplete: "new-password", spellcheck: "false", "aria-describedby": "model-token-state model-token-help" });
  const tokenState = h("span", { id: "model-token-state" });
  const cacheLocations = h("small", { class: "muted mono files" });
  const formMessage = h("p", { role: "status", "aria-live": "polite" });
  const runningNote = h("p", { class: "muted", hidden: true }, "다운로드 중에는 저장 위치와 토큰을 바꿀 수 없습니다.");
  const saveButton = h("button", { type: "submit" }, "저장");
  const clearButton = h("button", { type: "button", class: "danger", onclick: () => clearToken() }, "저장된 토큰 삭제");
  const fieldset = h(
    "fieldset",
    { class: "stack plain" },
    h(
      "div",
      { class: "field" },
      h("label", { for: "model-cache-dir" }, "모델 저장 위치"),
      cacheInput,
      h(
        "small",
        { id: "model-cache-help", class: "muted" },
        "서버의 절대 경로입니다. 바꿔도 기존 파일은 옮기지 않으며, 다음 다운로드와 다음 번역 작업부터 새 위치를 사용합니다.",
      ),
      cacheLocations,
    ),
    h(
      "div",
      { class: "field" },
      h("label", { for: "model-hf-token" }, "Hugging Face 토큰 (선택)"),
      tokenInput,
      h("small", { class: "muted" }, "상태: ", tokenState),
      h(
        "small",
        { id: "model-token-help", class: "muted" },
        "공개 모델은 토큰 없이 받을 수 있습니다. 비워 두고 저장하면 기존 토큰을 유지합니다. 여기서 저장한 토큰은 서버 환경의 토큰보다 우선하며, 암호화해 보관하고 화면에 다시 표시하지 않습니다.",
      ),
    ),
    h("div", { class: "row" }, saveButton, clearButton),
  );
  const form = h("form", { class: "stack" }, fieldset, runningNote, formMessage);

  const setFormMessage = (text, ok) => {
    formMessage.textContent = text;
    formMessage.className = ok ? "ok" : "error";
  };
  const saveSetup = async (json, successText) => {
    saveButton.disabled = clearButton.disabled = true;
    formMessage.textContent = "";
    try {
      const latest = await api("/api/model-setup", { method: "PUT", json });
      tokenInput.value = "";
      if (version !== state.routeVersion) return;
      cacheInput.value = latest.cache_dir || "";
      sync(latest);
      setFormMessage(successText, true);
    } catch (err) {
      if (version === state.routeVersion) setFormMessage(`저장하지 못했습니다: ${err.message}`, false);
    } finally {
      saveButton.disabled = false;
      clearButton.disabled = false;
    }
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const token = tokenInput.value.trim();
    const json = { cache_dir: cacheInput.value.trim() };
    if (token) json.token = token;
    saveSetup(json, token ? "저장했습니다. 토큰은 암호화해 보관했고 다시 표시하지 않습니다." : "저장했습니다.");
  });
  const clearToken = () => {
    if (!confirm("저장된 Hugging Face 토큰을 삭제할까요?\n삭제하면 이 앱은 서버 환경에 있는 토큰도 사용하지 않습니다. 이용 동의가 필요한 모델은 토큰을 다시 저장해야 받을 수 있습니다.")) return;
    saveSetup({ clear_token: true }, "저장된 토큰을 삭제했습니다.");
  };

  // --- download status (only the one-line summary is a live region, so
  // progress polling does not flood screen readers)
  const statusLine = h("p", { class: "status-line", role: "status", "aria-live": "polite" });
  const statusDetail = h("div", { class: "stack" });
  let announced = "";

  // --- profiles
  const profilesBox = h("div", { class: "stack" });

  const profileLabel = (id) => setup.profiles?.find((p) => p.id === id)?.label || id;

  const renderStatus = () => {
    const s = setup.status || { state: "idle", message: "" };
    const label = PREP_STATE[s.state] || s.state;
    const summary = s.profile_id ? `${label} · ${profileLabel(s.profile_id)}` : label;
    if (summary !== announced) {
      announced = summary;
      statusLine.replaceChildren(h("span", { class: `badge ${PREP_BADGE[s.state] || ""}` }, label), s.profile_id ? ` ${profileLabel(s.profile_id)}` : "");
    }
    const parts = [];
    if (s.state === "idle") {
      parts.push(h("p", { class: "muted" }, "아직 시작한 다운로드가 없습니다. 아래 프로필에서 받을 모델을 고르세요."));
    } else if (s.state === "running") {
      const counted = Number.isFinite(s.total) && s.total > 0;
      parts.push(
        counted
          ? h("progress", { max: s.total, value: s.done ?? 0, "aria-label": "모델 다운로드 진행률" })
          : h("progress", { "aria-label": "모델 다운로드 준비 중" }),
        h("p", { class: "muted" }, counted ? `파일 ${s.done ?? 0}/${s.total}` : "다운로드 준비 중…", s.current ? h("span", { class: "mono" }, ` · ${s.current}`) : ""),
      );
      if (s.message) parts.push(h("p", { class: "muted" }, s.message));
      parts.push(h("p", { class: "muted" }, "이 화면을 떠나도 서버에서 계속 진행됩니다. 큰 모델은 시간이 오래 걸릴 수 있습니다."));
    } else if (s.state === "done") {
      parts.push(h("p", { class: "ok" }, s.message || "모델 파일을 모두 받았습니다."));
      if (s.finished_at) parts.push(h("p", { class: "muted" }, `완료 시각: ${fmtTime(s.finished_at)}`));
    } else if (s.state === "failed") {
      parts.push(
        h("p", { class: "error" }, s.message || "다운로드에 실패했습니다."),
        h(
          "ul",
          { class: "muted hints" },
          h("li", {}, "서버의 인터넷 연결과 Hugging Face 접속 상태를 확인하세요."),
          h("li", {}, "저장 위치에 쓰기 권한과 충분한 여유 공간이 있는지 확인하세요."),
          h("li", {}, "이용 동의가 필요한 모델이면 Hugging Face 모델 페이지에서 동의한 계정의 토큰을 저장하세요."),
        ),
      );
      const failed = setup.profiles?.find((p) => p.id === s.profile_id);
      if (failed) parts.push(h("div", { class: "row" }, h("button", { onclick: () => prepare(failed) }, "다시 시도")));
    }
    statusDetail.replaceChildren(...parts);
  };

  const renderProfiles = () => {
    const running = isRunning();
    const groups = new Map();
    for (const profile of setup.profiles || []) {
      if (!groups.has(profile.engine)) groups.set(profile.engine, []);
      groups.get(profile.engine).push(profile);
    }
    if (!groups.size) {
      profilesBox.replaceChildren(h("p", { class: "muted" }, "서버가 알려 준 모델 프로필이 없습니다."));
      return;
    }
    profilesBox.replaceChildren(
      ...[...groups].map(([engineId, profiles]) => {
        const engine = engineMeta(engineId);
        const installed = profiles[0].engine_installed ?? engine?.available;
        return h(
          "section",
          { class: "engine-group stack", id: `models-${engineId}`, "data-engine": engineId, tabindex: "-1", "aria-labelledby": `models-${engineId}-title` },
          h(
            "div",
            { class: "row between" },
            h("h3", { id: `models-${engineId}-title` }, engine?.name || engineId),
            h("span", { class: `badge ${installed ? "done" : "failed"}` }, installed ? "엔진 설치됨" : "엔진 미설치"),
          ),
          installed
            ? null
            : h("p", { class: "muted" }, engine?.reason ? `${engine.reason} ` : "", "엔진은 설치 스크립트로 따로 설치해야 합니다. 모델만 받아도 엔진 없이는 번역할 수 없습니다."),
          profiles.some((p) => p.default)
            ? null
            : h("p", { class: "muted" }, "이 엔진은 일부 보조 모델만 여기서 미리 받을 수 있습니다. 번역을 실행할 때 엔진이 다른 모델을 추가로 내려받을 수 있습니다."),
          profiles.map((profile) => profileCard(profile, running)),
        );
      }),
    );
  };

  const profileCard = (profile, running) => {
    const s = setup.status || {};
    const active = s.profile_id === profile.id;
    const role = profileRole(profile);
    const badges = [
      h("span", { class: "badge" }, { default: "기본 구성", supplemental: "보조 모델", alternative: "추가 선택" }[role]),
      active && s.state === "running"
        ? h("span", { class: "badge running" }, "다운로드 진행 중")
        : profile.prepared
          ? h("span", { class: "badge done" }, "모델 준비됨")
          : active && s.state === "failed"
            ? h("span", { class: "badge failed" }, "다운로드 실패")
            : h("span", { class: "badge queued" }, "모델 미준비"),
    ];
    return h(
      "article",
      { class: `profile ${active ? "active" : ""}` },
      h("div", { class: "row between" }, h("h4", {}, profile.label || profile.id), h("div", { class: "row" }, badges)),
      profile.description ? h("p", { class: "muted" }, profile.description) : null,
      role === "alternative" ? h("p", { class: "muted" }, "이 모델을 받을 때 같은 엔진의 기본 구성도 함께 확인하고 없으면 받습니다.") : null,
      h(
        "ul",
        { class: "models" },
        (profile.models || []).map((model) =>
          h(
            "li",
            {},
            h("a", { href: hfModelUrl(model), target: "_blank", rel: "noopener noreferrer", class: "mono" }, model.repo),
            model.revision ? h("small", { class: "muted mono" }, ` @ ${model.revision}`) : null,
            h("div", { class: "muted mono files" }, `파일: ${fileSummary(model.files, 8)}`),
            model.note ? h("div", { class: "muted" }, model.note) : null,
          ),
        ),
      ),
      profile.missing?.length && !profile.prepared
        ? h("p", { class: "muted mono files" }, `없는 파일: ${fileSummary(profile.missing, 5)}`)
        : null,
      h(
        "div",
        { class: "row" },
        h(
          "button",
          { class: profile.prepared ? "secondary" : "", disabled: running, "aria-label": `${profile.label || profile.id} ${profile.prepared ? "다시 받기" : "모델 다운로드"}`, onclick: () => prepare(profile) },
          profile.prepared ? "다시 받기" : "모델 다운로드",
        ),
      ),
    );
  };

  const profileRole = (profile) => {
    if (profile.default) return "default";
    const engineHasDefault = (setup.profiles || []).some((p) => p.engine === profile.engine && p.default);
    return profile.supplemental || !engineHasDefault ? "supplemental" : "alternative";
  };

  const prepare = async (profile) => {
    if (isRunning()) return;
    const bundled =
      profileRole(profile) === "alternative"
        ? (setup.profiles || []).filter((p) => p.engine === profile.engine && p.default && !p.prepared)
        : [];
    // Backend may already list the base models inside the alternative bundle.
    const lines = [...new Set([profile, ...bundled].flatMap((p) => p.models || []).map((m) => `• ${m.repo} — ${fileSummary(m.files)}`))].join("\n");
    const ok = confirm(
      `"${profile.label || profile.id}" 모델을 Hugging Face에서 내려받을까요?` +
        (bundled.length ? " 아직 준비되지 않은 기본 구성도 함께 받습니다." : "") +
        `\n\n${lines}\n\n저장 위치: ${setup.cache_dir}\n\n` +
        "모델 파일은 클 수 있어 네트워크 사용량과 디스크 공간이 필요합니다(정확한 크기는 모델 페이지에서 확인). 모델을 받아도 GPU 추론 성능이나 동작을 보장하지는 않습니다.",
    );
    if (!ok) return;
    try {
      const status = await api("/api/model-setup/prepare", { method: "POST", json: { profile_id: profile.id } });
      if (version !== state.routeVersion) return;
      setup = { ...setup, status: status?.state ? status : { state: "running", message: "", profile_id: profile.id } };
      sync(setup);
    } catch (err) {
      if (version !== state.routeVersion) return;
      toast(`다운로드를 시작하지 못했습니다: ${err.message}`, true);
      poll();
    }
  };

  const renderTokenState = () => {
    const source = setup.token_source ?? (setup.token_configured ? "saved" : null);
    const external = source === "env" || source === "file";
    tokenState.textContent = external ? "서버 환경에서 관리 중 (이 화면에서 삭제 불가)" : setup.token_configured ? "저장됨" : "없음";
    tokenInput.placeholder = setup.token_configured ? "바꿀 때만 입력" : "hf_로 시작하는 토큰";
    clearButton.hidden = !(setup.token_configured && !external);
  };

  const schedulePoll = (delay) => {
    stopPolling();
    state.pollTimer = setTimeout(poll, delay);
  };
  const poll = async () => {
    try {
      const latest = await api("/api/model-setup");
      if (version !== state.routeVersion) return;
      sync(latest);
    } catch (err) {
      if (version !== state.routeVersion) return;
      statusLine.replaceChildren(h("span", { class: "error" }, `상태를 불러오지 못했습니다: ${err.message} — 잠시 후 다시 확인합니다.`));
      announced = "";
      schedulePoll(5000);
    }
  };

  let profilesKey = null;
  function sync(latest) {
    setup = latest;
    const running = isRunning();
    fieldset.disabled = running;
    runningNote.hidden = !running;
    cacheLocations.textContent = `실제 사용 위치 · Koharu: ${setup.koharu_cache_dir || ""} · Hugging Face: ${setup.hf_cache_dir || ""}`;
    renderTokenState();
    renderStatus();
    // Rebuild profile cards only when something they show changed, so polling
    // does not reset keyboard focus inside the list.
    const key = JSON.stringify([setup.status?.state, setup.status?.profile_id, setup.profiles]);
    if (key !== profilesKey) {
      profilesKey = key;
      renderProfiles();
    }
    if (running) schedulePoll(2000);
  }

  view.replaceChildren(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "모델 준비"),
      h(
        "p",
        {},
        "엔진 설치와 모델 다운로드는 별도 단계입니다. 설치 스크립트는 엔진만 설치하고 모델은 받지 않습니다. 이 화면에서 필요한 모델을 고르고 직접 다운로드를 시작할 때만 받습니다.",
      ),
    ),
    h("section", { class: "card narrow" }, h("h3", {}, "저장 위치와 토큰"), form),
    h("section", { class: "card", "aria-labelledby": "model-status-title" }, h("h3", { id: "model-status-title" }, "다운로드 상태"), statusLine, statusDetail),
    h("section", { class: "card stack", "aria-labelledby": "model-profiles-title" }, h("h3", { id: "model-profiles-title" }, "모델 프로필"), profilesBox),
  );
  sync(setup);
  if (focusEngine) {
    const target = [...profilesBox.querySelectorAll(".engine-group")].find((el) => el.dataset.engine === focusEngine);
    if (target) {
      target.scrollIntoView({ block: "start" });
      target.focus({ preventScroll: true });
    }
  }
}

// ------------------------------------------------------------------ providers
function renderProviders() {
  const kinds = state.meta.provider_kinds;
  const list = state.providers.map((p) => providerCard(p));
  const kindSelect = h("select", { name: "kind" }, kinds.map((k) => h("option", { value: k.kind }, k.label)));
  const addForm = h(
    "form",
    { class: "row", onsubmit: (e) => { e.preventDefault(); editProvider({ kind: kindSelect.value }); } },
    kindSelect,
    h("button", { type: "submit" }, "제공자 추가"),
  );
  view.replaceChildren(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "LLM 제공자"),
      h(
        "p",
        { class: "muted" },
        "OpenAI(ChatGPT 구독)와 xAI(SuperGrok)는 OAuth 로그인을 지원합니다. ChatGPT 로그인은 Codex CLI 방식을 이용하는 비공식 경로라 OpenAI 정책 변경 시 동작하지 않을 수 있습니다. Anthropic은 약관상 외부 앱의 Claude 계정 로그인을 금지하므로 API 키만 지원합니다.",
      ),
      addForm,
    ),
    ...list,
  );
}

function providerCard(p) {
  const preset = state.meta.provider_kinds.find((k) => k.kind === p.kind);
  const status = h("div", { class: "muted" });
  const buttons = h(
    "div",
    { class: "row" },
    h("button", { class: "secondary", onclick: () => editProvider(p) }, "수정"),
    p.auth === "oauth"
      ? p.connected
        ? h("button", { class: "secondary", onclick: async () => { await api(`/api/providers/${p.id}/logout`, { method: "POST" }); route(); } }, "로그아웃")
        : h("button", { onclick: () => startOAuth(p, status) }, preset.kind === "openai" ? "ChatGPT로 로그인" : "xAI로 로그인")
      : null,
    h("button", { class: "secondary", disabled: !p.connected, onclick: () => testProvider(p, status) }, "연결 테스트"),
    h("button", { class: "danger", onclick: async () => { if (confirm(`${p.name} 제공자를 삭제할까요?`)) { await api(`/api/providers/${p.id}`, { method: "DELETE" }); route(); } } }, "삭제"),
  );
  return h(
    "section",
    { class: "card" },
    h(
      "div",
      { class: "row between" },
      h("h3", {}, `${p.name} `, h("span", { class: `badge ${p.connected ? "done" : "failed"}` }, p.connected ? "연결됨" : "미연결")),
      buttons,
    ),
    h(
      "p",
      {},
      `${preset.label} · ${authLabel(p.auth)} · 모델 ${p.model}`,
      p.base_url ? ` · ${p.base_url}` : "",
      p.account ? ` · ${p.account}` : "",
      p.vision ? " · 비전" : "",
    ),
    status,
  );
}

function authLabel(auth) {
  return { oauth: "OAuth 로그인", api_key: "API 키", none: "인증 없음" }[auth] || auth;
}

async function testProvider(p, status) {
  status.textContent = "테스트 중…";
  const result = await api(`/api/providers/${p.id}/test`, { method: "POST" });
  status.textContent = result.ok ? `응답 (${result.seconds}s): ${result.reply}` : `실패: ${result.error}`;
  status.className = result.ok ? "ok" : "error";
}

async function startOAuth(p, status) {
  const session = await api(`/api/providers/${p.id}/oauth`, { method: "POST" });
  const poll = async () => {
    const s = await api(`/api/oauth/${session.id}`);
    if (s.status === "pending" || s.status === "starting") {
      status.replaceChildren(
        s.user_code
          ? h(
              "div",
              { class: "oauth" },
              h("p", {}, "아래 주소에서 코드를 입력하고 승인하세요."),
              h("p", {}, h("a", { href: s.verification_url, target: "_blank", rel: "noopener" }, s.verification_url)),
              h("p", { class: "code" }, s.user_code),
              h("button", { class: "secondary", onclick: () => api(`/api/oauth/${session.id}/cancel`, { method: "POST" }) }, "취소"),
            )
          : "로그인 준비 중…",
      );
      setTimeout(poll, 2000);
    } else if (s.status === "done") {
      toast(`로그인 완료${s.account ? `: ${s.account}` : ""}`);
      route();
    } else {
      status.textContent = s.status === "cancelled" ? "취소되었습니다." : `로그인 실패: ${s.message}`;
      status.className = "error";
    }
  };
  poll();
}

function editProvider(p) {
  const preset = state.meta.provider_kinds.find((k) => k.kind === p.kind);
  const isNew = !p.id;
  const values = {
    name: p.name ?? preset.label,
    auth: p.auth ?? preset.auth[0],
    base_url: p.base_url ?? preset.base_url ?? "",
    model: p.model ?? preset.model,
    extra_body: JSON.stringify(p.extra_body ?? preset.extra_body ?? {}, null, 2),
    vision: p.vision ?? false,
  };
  const authSelect = h("select", { name: "auth" }, preset.auth.map((a) => h("option", { value: a, selected: a === values.auth }, authLabel(a))));
  const keyField = h("label", {}, "API 키", h("input", { name: "api_key", type: "password", autocomplete: "off", placeholder: p.connected && p.auth === "api_key" ? "(저장됨 — 바꿀 때만 입력)" : "" }));
  const baseField = h(
    "label",
    {},
    p.kind === "openai_compatible" ? "Base URL (필수)" : "Base URL (비우면 기본값)",
    h("input", { name: "base_url", value: values.base_url, placeholder: preset.base_url || "" }),
  );
  // OAuth uses fixed endpoints and no key; "none" needs no key either.
  const syncAuth = () => {
    keyField.hidden = authSelect.value !== "api_key";
    baseField.hidden = authSelect.value === "oauth";
  };
  authSelect.addEventListener("change", syncAuth);
  const form = h(
    "form",
    { class: "stack" },
    h("h2", {}, `${preset.label} ${isNew ? "추가" : "수정"}`),
    h("label", {}, "이름", h("input", { name: "name", value: values.name, required: true })),
    h("label", {}, "인증 방식", authSelect),
    keyField,
    baseField,
    h("label", {}, "모델", h("input", { name: "model", value: values.model, required: true })),
    h("label", { class: "check" }, h("input", { type: "checkbox", name: "vision", checked: values.vision }), " 이미지 입력 허용 (비전 모델)"),
    h(
      "label",
      {},
      "추가 요청 필드 (JSON, 요청 본문에 병합)",
      h("textarea", { name: "extra_body", rows: 5, class: "mono" }, values.extra_body),
      h("small", { class: "muted" }, '로컬 DeepSeek 예: {"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}, "max_tokens": 16384}'),
    ),
    h("div", { class: "row" }, h("button", { type: "submit" }, "저장"), h("button", { type: "button", class: "secondary", onclick: () => route() }, "닫기")),
  );
  syncAuth();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form);
    try {
      await api("/api/providers", {
        method: "POST",
        json: {
          id: p.id,
          kind: p.kind,
          name: data.get("name"),
          auth: data.get("auth"),
          api_key: data.get("api_key") || undefined,
          base_url: data.get("base_url"),
          model: data.get("model"),
          vision: data.get("vision") === "on",
          extra_body: data.get("extra_body"),
        },
      });
      toast("저장했습니다.");
      route();
    } catch (err) {
      toast(err.message, true);
    }
  });
  view.replaceChildren(h("section", { class: "card narrow" }, form));
}

window.addEventListener("hashchange", route);
route();
