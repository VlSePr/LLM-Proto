/* Minimal slide engine: 1920×1080 stage, fragments, hash routing, overview, speaker notes. */
(function () {
  const stage = document.getElementById("stage");
  const slides = Array.from(stage.querySelectorAll(":scope > .slide"));
  const progress = document.getElementById("progress");
  const total = slides.length;
  let cur = 0;
  let notesWin = null;

  // chrome on every slide
  slides.forEach((s, i) => {
    s.dataset.index = i;
    const dark = s.classList.contains("dark");
    s.insertAdjacentHTML("beforeend",
      `<img class="chrome-brand" src="assets/logo/${dark ? "image2" : "image10"}.svg" alt="EPAM.AI Conference">` +
      `<img class="chrome-epam" src="assets/logo/${dark ? "image3" : "image11"}.svg" alt="epam">` +
      `<div class="chrome-foot"><img src="assets/logo/image12.svg" alt="EPAM.AI CONFERENCE"><span>${i + 1}</span></div>`);
  });

  function fitCentered() {
    if (document.body.classList.contains("overview")) return;
    const k = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    stage.style.left = `${(window.innerWidth - 1920 * k) / 2}px`;
    stage.style.top = `${(window.innerHeight - 1080 * k) / 2}px`;
    stage.style.transform = `scale(${k})`;
  }

  const steps = (s) => Array.from(s.querySelectorAll(".step"));

  function show(i, stepMode) {
    i = Math.max(0, Math.min(total - 1, i));
    const prev = slides[cur];
    slides.forEach((s, j) => s.classList.toggle("active", j === i));
    const st = steps(slides[i]);
    st.forEach((el) => el.classList.toggle("shown", stepMode === "all"));
    if (prev !== slides[i]) prev.dispatchEvent(new CustomEvent("slide:leave"));
    cur = i;
    progress.style.width = `${((i + 1) / total) * 100}%`;
    history.replaceState(null, "", `#/${i + 1}`);
    slides[i].dispatchEvent(new CustomEvent("slide:enter"));
    syncNotes();
  }

  function next() {
    const hidden = steps(slides[cur]).filter((e) => !e.classList.contains("shown"));
    if (hidden.length) { hidden[0].classList.add("shown"); hidden[0].dispatchEvent(new CustomEvent("step:shown", { bubbles: true })); syncNotes(); return; }
    if (cur < total - 1) show(cur + 1);
  }
  function prev() {
    const shown = steps(slides[cur]).filter((e) => e.classList.contains("shown"));
    if (shown.length) { shown[shown.length - 1].classList.remove("shown"); syncNotes(); return; }
    if (cur > 0) show(cur - 1, "all");
  }

  function toggleOverview(force) {
    const on = force ?? !document.body.classList.contains("overview");
    document.body.classList.toggle("overview", on);
    if (on) { stage.style.transform = ""; stage.style.left = ""; stage.style.top = ""; slides[cur].scrollIntoView({ block: "center" }); }
    else fitCentered();
  }

  function fullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  // ─── Speaker notes window ───
  function openNotes() {
    notesWin = window.open("", "llm-notes", "width=980,height=720");
    if (!notesWin) return;
    notesWin.document.write(`<!doctype html><html><head><title>Speaker notes</title><style>
      body{font:20px/1.5 "Source Sans 3","Segoe UI",sans-serif;margin:0;background:#120a26;color:#eee;display:flex;flex-direction:column;height:100vh}
      header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;background:linear-gradient(90deg,#3D00F3,#6042FF,#FF4B9B,#FF7701);color:#fff;font-weight:700}
      #clock{font:700 28px Consolas,monospace}
      main{flex:1;overflow:auto;padding:22px 28px}
      h2{margin:0 0 8px;font-size:30px} .nx{color:#b9b0dd;font-size:17px;margin-bottom:18px}
      .notes p{margin:0 0 12px} .notes li{margin-bottom:8px} .steps{color:#ff9dc9;font-weight:700}
      .n-time{font-size:15px;color:#b9b0dd;margin-bottom:12px;letter-spacing:.04em} .n-time b{color:#ffb36b}
      .n-say{font-size:23px;line-height:1.55;color:#fff;border-left:4px solid #ff4b9b;padding-left:16px}
      .n-say b{color:#ffd1e6}
      .n-keys{font-size:17px;color:#d9d3f5;background:rgba(255,255,255,.05);border-radius:10px;padding:12px 16px 12px 34px;margin:6px 0 14px}
      .n-do{font-size:17px;color:#b8f5cf} .n-ask{font-size:17px;color:#ffd79a}
      footer{padding:10px 22px;color:#8f86b8;font-size:15px} button{font:inherit;padding:6px 14px;margin-left:6px;border-radius:8px;border:0;cursor:pointer}
    </style></head><body><header><span id="pos"></span><span id="clock">00:00</span><span><button id="p">◀</button><button id="n">▶</button><button id="r">reset timer</button></span></header>
    <main><h2 id="t"></h2><div class="nx" id="nx"></div><div class="steps" id="st"></div><div class="notes" id="notes"></div></main>
    <footer>Keys in this window: ← / → move the deck.</footer></body></html>`);
    notesWin.document.close();
    const d = notesWin.document;
    let t0 = Date.now();
    d.getElementById("p").onclick = prev; d.getElementById("n").onclick = next;
    d.getElementById("r").onclick = () => { t0 = Date.now(); };
    d.addEventListener("keydown", onKey);
    setInterval(() => {
      if (!notesWin || notesWin.closed) return;
      const s = Math.floor((Date.now() - t0) / 1000);
      d.getElementById("clock").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    }, 500);
    syncNotes();
  }
  // ─── Full script: every slide's notes in order, printable (P) ───
  function openScript() {
    const w = window.open("", "llm-script");
    if (!w) return;
    const body = slides.map((s, i) => {
      const n = s.querySelector("aside.notes");
      return `<section><h2><span>${i + 1}</span>${titleOf(s)}</h2>${n ? n.innerHTML : "<p><i>No notes</i></p>"}</section>`;
    }).join("");
    w.document.write(`<!doctype html><html><head><title>Speaker script</title><style>
      body{font:17px/1.5 "Source Sans 3","Segoe UI",sans-serif;max-width:900px;margin:0 auto;padding:30px;color:#222}
      h1{font-size:30px;margin:0 0 6px} .hint{color:#777;margin-bottom:24px}
      section{border-top:2px solid #6042FF;padding:14px 0 6px;break-inside:avoid-page}
      h2{font-size:22px;margin:0 0 8px} h2 span{display:inline-block;min-width:34px;color:#FF4B9B}
      .n-time{font-size:13px;color:#777;margin-bottom:6px} .n-time b{color:#c25a00}
      .n-say{font-size:17px;border-left:3px solid #FF4B9B;padding-left:12px;margin:0 0 10px}
      .n-keys{font-size:14px;color:#444;background:#f6f4fc;border-radius:8px;padding:8px 12px 8px 30px}
      .n-do{font-size:14px;color:#166534} .n-ask{font-size:14px;color:#9a5b00}
      button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid #6042FF;background:#fff;color:#6042FF;cursor:pointer}
      @media print{button,.hint{display:none} body{padding:0}}
    </style></head><body><h1>Building an LLM from scratch · speaker script</h1>
    <div class="hint">${slides.length} slides · <button onclick="print()">Print / save as PDF</button></div>${body}</body></html>`);
    w.document.close();
  }

  const titleOf = (s) => (s.dataset.title || s.querySelector("h1,h2")?.textContent || "").trim();
  function syncNotes() {
    if (!notesWin || notesWin.closed) return;
    const d = notesWin.document, s = slides[cur];
    const st = steps(s), shown = st.filter((e) => e.classList.contains("shown")).length;
    d.getElementById("pos").textContent = `Slide ${cur + 1} / ${total}`;
    d.getElementById("t").textContent = titleOf(s);
    d.getElementById("nx").textContent = cur < total - 1 ? `Next: ${titleOf(slides[cur + 1])}` : "Last slide";
    d.getElementById("st").textContent = st.length ? `Reveal ${shown} / ${st.length}` : "";
    const n = s.querySelector("aside.notes");
    d.getElementById("notes").innerHTML = n ? n.innerHTML : "<i>No notes</i>";
  }

  function onKey(e) {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) && !["Escape"].includes(e.key)) return;
    if (document.getElementById("help").classList.contains("open") && e.key !== "?") { document.getElementById("help").classList.remove("open"); return; }
    switch (e.key) {
      case "ArrowRight": case "PageDown": case " ": case "ArrowDown": e.preventDefault(); next(); break;
      case "ArrowLeft": case "PageUp": case "ArrowUp": e.preventDefault(); prev(); break;
      case "Home": show(0); break;
      case "End": show(total - 1, "all"); break;
      case "o": case "O": case "Escape": toggleOverview(e.key === "Escape" ? false : undefined); break;
      case "f": case "F": fullscreen(); break;
      case "s": case "S": openNotes(); break;
      case "p": case "P": openScript(); break;
      case "?": document.getElementById("help").classList.toggle("open"); break;
      default: return;
    }
  }
  document.addEventListener("keydown", onKey);

  stage.addEventListener("click", (e) => {
    if (!document.body.classList.contains("overview")) return;
    const s = e.target.closest(".slide");
    if (s) { toggleOverview(false); show(+s.dataset.index, "all"); }
  });

  // touch swipe
  let tx = null;
  document.addEventListener("touchstart", (e) => { tx = e.touches[0].clientX; }, { passive: true });
  document.addEventListener("touchend", (e) => {
    if (tx === null) return;
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 60) (dx < 0 ? next : prev)();
    tx = null;
  });

  document.getElementById("hud").addEventListener("click", (e) => {
    const a = e.target.dataset.act;
    if (a === "prev") prev(); else if (a === "next") next(); else if (a === "ov") toggleOverview();
    else if (a === "fs") fullscreen(); else if (a === "notes") openNotes(); else if (a === "script") openScript(); else if (a === "help") document.getElementById("help").classList.toggle("open");
  });

  window.addEventListener("resize", fitCentered);
  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/#\/(\d+)/);
    if (m && +m[1] - 1 !== cur) show(+m[1] - 1, "all");
  });

  fitCentered();
  const m = location.hash.match(/#\/(\d+)/);
  show(m ? +m[1] - 1 : 0, m ? "all" : undefined);
  window.deck = { show, next, prev, slides, get current() { return cur; } };
})();
