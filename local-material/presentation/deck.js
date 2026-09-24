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
    else if (a === "fs") fullscreen(); else if (a === "notes") openNotes(); else if (a === "help") document.getElementById("help").classList.toggle("open");
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
