/* Interactive widgets for the deck. Each one initialises lazily on its slide's first "slide:enter". */
(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  const fmt = (n, d = 1) => Number(n).toFixed(d);

  function onEnter(id, init) {
    const el = document.getElementById(id);
    if (!el) return;
    const slide = el.closest(".slide");
    let done = false;
    const run = () => { if (!done) { done = true; init(el, slide); } };
    slide.addEventListener("slide:enter", run);
    if (slide.classList.contains("active")) run();
  }

  // ════════════════════════════════════════════════════════════
  // 1. Architecture explorer (src/model.py, preset "large")
  // ════════════════════════════════════════════════════════════
  const D = 2048, L = 32, H = 32, KV = 8, HD = 64, F = 5632, V = 32000;
  const M = (x) => `${fmt(x / 1e6, x < 1e7 ? 2 : 1)}M`;
  const ARCH = [
    { id: "emb", cls: "io", name: "Token embedding", tag: "tok_emb", shape: `ids (B, T) → x (B, T, ${D})`, params: V * D,
      formula: "x = E[ids]", text: `A 32,000 × 2048 lookup table. It is <b>tied</b> with the LM head (<code>tie_embeddings=True</code>), so these 65.5M weights are used twice, once to read tokens in and once to score the next token.` },
    { id: "n1", cls: "norm", name: "RMSNorm", tag: "attn_norm", shape: `(B, T, ${D}) → same`, params: D, block: true,
      formula: "y = x / √(mean(x²) + ε) · g", text: `Pre-norm: normalise <i>before</i> each sub-layer, so the residual stream itself is never normalised. RMSNorm drops LayerNorm's mean-centring and bias, which makes it cheaper and just as stable. ε = 1e-5.` },
    { id: "attn", cls: "attn", name: "Grouped-query attention", tag: "attn.wq/wk/wv/wo", shape: `Q (B, 32, T, 64) · K,V (B, 8, T, 64)`, params: D * D * 2 + 2 * KV * HD * D, block: true,
      formula: "softmax(QKᵀ/√64 + causal) · V", text: `32 query heads share <b>8</b> key/value heads, 4 query heads per KV group. <code>wk</code> and <code>wv</code> are only 512 × 2048. That cuts the KV cache by 4× compared with full multi-head attention, at almost no quality cost. PyTorch SDPA runs the fused flash kernel.` },
    { id: "rope", cls: "attn", name: "RoPE (inside attention)", tag: "no params", shape: `rotate (q, k) pairs by m·θᵢ`, params: 0, block: true,
      formula: "θᵢ = 10000^(−2i/64)", text: `Rotary position embeddings rotate each 2-D pair of q/k dimensions by an angle that grows with position. The q·k dot product then depends only on the <b>relative</b> distance m − n. It adds no parameters and extrapolates gracefully.` },
    { id: "r1", cls: "res", name: "+ residual", tag: "x = x + attn(norm(x))", shape: "", params: 0, block: true,
      formula: "x ← x + Attn(RMSNorm(x))", text: `The residual stream is the model's working memory. Every sub-layer <i>adds</i> to it and never overwrites it. In the activation-stats chart later on, its std grows from ~3 at layer 0 to ~24 at layer 31.` },
    { id: "n2", cls: "norm", name: "RMSNorm", tag: "ffn_norm", shape: `(B, T, ${D}) → same`, params: D, block: true,
      formula: "y = x / √(mean(x²) + ε) · g", text: `The second pre-norm of the block. Its learned gain <code>g</code> shrinks in the last layers (see the weight-stats chart), so the model learns to turn deep FFN contributions down.` },
    { id: "ffn", cls: "ffn", name: "SwiGLU feed-forward", tag: "ffn.w_gate/w_up/w_down", shape: `${D} → ${F} → ${D}`, params: 3 * D * F, block: true,
      formula: "W_down( SiLU(W_gate x) ⊙ W_up x )", text: `A gated FFN with three matrices instead of two. The hidden size 5632 ≈ 8/3 · 2048, rounded, keeps the parameter count equal to a classic 4× MLP. Per layer it holds ~77% of the parameters, and this is the module MoE later replaces.` },
    { id: "r2", cls: "res", name: "+ residual", tag: "x = x + ffn(norm(x))", shape: "", params: 0, block: true,
      formula: "x ← x + FFN(RMSNorm(x))", text: `After 32 of these blocks the stream carries everything the model "knows" about the next token.` },
    { id: "nf", cls: "norm", name: "Final RMSNorm", tag: "norm", shape: `(B, T, ${D})`, params: D,
      formula: "y = RMSNorm(x)", text: `The last normalisation before the output projection.` },
    { id: "head", cls: "io", name: "LM head → logits", tag: "output (tied)", shape: `(B, T, ${D}) → (B, T, 32000)`, params: 0,
      formula: "p(next) = softmax(x · Eᵀ)", text: `This reuses the embedding matrix, so it adds 0 new parameters. Cross-entropy on these logits is the <i>only</i> training signal.` },
  ];
  const perLayer = ARCH.filter((a) => a.block).reduce((s, a) => s + a.params, 0);

  onEnter("arch", (el) => {
    const stack = $(".arch-stack", el), det = $(".arch-detail", el);
    const node = (a) => `<div class="arch-node ${a.cls}" data-id="${a.id}"><span>${a.name}</span><small>${a.params ? M(a.params) : a.tag}</small></div>`;
    stack.innerHTML =
      node(ARCH[0]) +
      `<div class="arch-block">${ARCH.filter((a) => a.block).map(node).join("")}<div class="x32 gtext">×32</div></div>` +
      ARCH.slice(-2).map(node).join("");
    const select = (id) => {
      const a = ARCH.find((x) => x.id === id);
      stack.querySelectorAll(".arch-node").forEach((n) => n.classList.toggle("sel", n.dataset.id === id));
      det.innerHTML = `
        <span class="eyebrow">${a.block ? "Inside each of the 32 blocks" : "Model input / output"}</span>
        <h3>${a.name}</h3>
        <div class="shape">${esc(a.shape || a.tag)}</div>
        <div class="math">${a.formula}</div>
        <p class="body">${a.text}</p>
        <div class="params">
          <div><b>${a.params ? M(a.params) : "0"}</b>params here</div>
          ${a.block ? `<div><b>${M(a.params * L)}</b>across 32 layers</div>` : ""}
          <div><b>1,508.5M</b>whole model</div>
        </div>`;
    };
    stack.addEventListener("click", (e) => { const n = e.target.closest(".arch-node"); if (n) select(n.dataset.id); });
    select("attn");
    $(".arch-trace", el).addEventListener("click", () => {
      const nodes = Array.from(stack.querySelectorAll(".arch-node"));
      nodes.forEach((n, i) => {
        setTimeout(() => { n.classList.add("pulse"); select(n.dataset.id); }, i * 650);
        setTimeout(() => n.classList.remove("pulse"), i * 650 + 900);
      });
    });
    $(".arch-perlayer", el).textContent = M(perLayer);
  });

  // ════════════════════════════════════════════════════════════
  // 2. RoPE: rotate a (q, k) pair; the dot product depends only on m − n
  // ════════════════════════════════════════════════════════════
  onEnter("rope", (el) => {
    const svg = $("svg", el), R = 220, cx = 300, cy = 280;
    const q0 = Math.PI / 7, k0 = -Math.PI / 5;  // base angles of the un-rotated vectors
    const inputs = { m: $("#rope-m", el), n: $("#rope-n", el), i: $("#rope-i", el) };
    const out = (k) => $(`output[for=rope-${k}]`, el);
    const theta = () => Math.pow(10000, (-2 * +inputs.i.value) / 64);
    const arrow = (ang, color, label, len = R) => {
      const x = cx + len * Math.cos(ang), y = cy - len * Math.sin(ang);
      return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${color}" stroke-width="9" stroke-linecap="round" marker-end="url(#ah-${color.slice(1)})"/>
              <text x="${cx + (len + 46) * Math.cos(ang)}" y="${cy - (len + 46) * Math.sin(ang) + 12}" fill="${color}" font-size="34" font-weight="700" text-anchor="middle" font-family="JetBrains Mono">${label}</text>`;
    };
    function draw() {
      const m = +inputs.m.value, n = +inputs.n.value, th = theta();
      const aq = q0 + m * th, ak = k0 + n * th;
      out("m").textContent = m; out("n").textContent = n; out("i").textContent = inputs.i.value;
      const arc = (a0, a1, r, c) => {
        const large = Math.abs(a1 - a0) % (2 * Math.PI) > Math.PI ? 1 : 0;
        return `<path d="M ${cx + r * Math.cos(a0)} ${cy - r * Math.sin(a0)} A ${r} ${r} 0 ${large} ${a1 > a0 ? 0 : 1} ${cx + r * Math.cos(a1)} ${cy - r * Math.sin(a1)}" fill="none" stroke="${c}" stroke-width="4" stroke-dasharray="8 7"/>`;
      };
      svg.innerHTML = `
        <defs>${["6042FF", "FF4B9B"].map((c) => `<marker id="ah-${c}" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4 z" fill="#${c}"/></marker>`).join("")}</defs>
        <circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="#e3dcf7" stroke-width="3"/>
        <line x1="${cx - R - 30}" y1="${cy}" x2="${cx + R + 30}" y2="${cy}" stroke="#eee8fb" stroke-width="2"/>
        <line x1="${cx}" y1="${cy - R - 30}" x2="${cx}" y2="${cy + R + 30}" stroke="#eee8fb" stroke-width="2"/>
        ${arc(aq, ak, 90, "#FF7701")}
        ${arrow(aq, "#6042FF", "q")}
        ${arrow(ak, "#FF4B9B", "k")}
        <circle cx="${cx}" cy="${cy}" r="9" fill="#222"/>`;
      const rel = ((aq - ak) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      $(".rope-read", el).innerHTML =
        `θ<sub>${inputs.i.value}</sub> = ${th.toExponential(2)} rad / token<br>` +
        `q rotated by m·θ = ${fmt(m * th, 3)} rad<br>` +
        `k rotated by n·θ = ${fmt(n * th, 3)} rad<br>` +
        `<b>relative angle</b> = (m−n)·θ + const = <b style="color:#FF7701">${fmt(rel, 3)}</b><br>` +
        `<b>q·k</b> ∝ cos(…) = <b class="gtext" style="font-size:34px">${fmt(Math.cos(aq - ak), 4)}</b>`;
    }
    Object.values(inputs).forEach((inp) => inp.addEventListener("input", draw));
    $(".rope-shift", el).addEventListener("click", () => {
      let k = 0;
      const t = setInterval(() => {
        inputs.m.value = Math.min(+inputs.m.max, +inputs.m.value + 1);
        inputs.n.value = Math.min(+inputs.n.max, +inputs.n.value + 1);
        draw();
        if (++k >= 12) clearInterval(t);
      }, 120);
    });
    draw();
  });

  // ════════════════════════════════════════════════════════════
  // 3. GQA: heads + KV-cache calculator
  // ════════════════════════════════════════════════════════════
  onEnter("gqa", (el) => {
    let kv = 8;
    const qRow = $(".gqa-q-row", el), kvRow = $(".gqa-kv-row", el), links = $(".gqa-links", el);
    qRow.innerHTML = Array.from({ length: H }, () => `<div class="gqa-q"></div>`).join("");
    kvRow.innerHTML = Array.from({ length: H }, () => `<div class="gqa-kv"></div>`).join("");
    const ctx = $("#gqa-ctx", el), bs = $("#gqa-bs", el);
    function draw() {
      const group = H / kv;
      // which KV slots are "real": the first of each group, drawn centred under its group
      const kvBoxes = kvRow.children;
      const kvIdx = Array.from({ length: kv }, (_, g) => Math.floor(g * group + (group - 1) / 2));
      Array.from(kvBoxes).forEach((b, i) => b.classList.toggle("off", !kvIdx.includes(i)));
      const W = links.clientWidth || 1680, cw = W / H;
      links.setAttribute("viewBox", `0 0 ${W} 110`);
      const colors = ["#6042FF", "#893FFF", "#C545CC", "#FF4B9B", "#F26B43", "#FF7701", "#3D00F3", "#e0529c"];
      links.innerHTML = Array.from({ length: H }, (_, q) => {
        const g = Math.floor(q / group), x1 = (q + 0.5) * cw, x2 = (kvIdx[g] + 0.5) * cw;
        return `<path d="M${x1},0 C${x1},60 ${x2},50 ${x2},110" stroke="${colors[g % colors.length]}" stroke-width="3" fill="none" opacity=".75"/>`;
      }).join("");
      const T = +ctx.value, B = +bs.value;
      $("output[for=gqa-ctx]", el).textContent = T.toLocaleString();
      $("output[for=gqa-bs]", el).textContent = B;
      const bytes = (k) => 2 * L * k * HD * T * B * 2; // K and V · layers · kv heads · head_dim · tokens · batch · bf16
      const gb = (b) => b / 1024 ** 3;
      const mx = bytes(32);
      [[32, "mha"], [8, "gqa"], [1, "mqa"]].forEach(([k, id]) => {
        $(`.m-${id} span`, el).style.width = `${(bytes(k) / mx) * 100}%`;
        $(`.v-${id}`, el).textContent = gb(bytes(k)) >= 1 ? `${fmt(gb(bytes(k)), 2)} GB` : `${fmt(bytes(k) / 1024 ** 2, 0)} MB`;
      });
      $(".gqa-pertok", el).textContent = `${fmt((2 * L * kv * HD * 2) / 1024, 0)} KiB per token`;
      el.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", +b.dataset.kv === kv));
      $(".gqa-desc", el).innerHTML = kv === 32 ? "<b>Multi-head:</b> every query head has its own K and V. Best quality, biggest cache."
        : kv === 1 ? "<b>Multi-query:</b> one K/V for all heads. Tiny cache, but quality drops."
        : "<b>Grouped-query (ours):</b> 4 query heads share one K/V head. Close to MHA quality with ¼ of the cache.";
    }
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { kv = +b.dataset.kv; draw(); }));
    [ctx, bs].forEach((i) => i.addEventListener("input", draw));
    draw();
    window.addEventListener("resize", draw);
  });

  // ════════════════════════════════════════════════════════════
  // 4. Sampling playground (mirrors src/generate.py order: rep-penalty → temperature → top-k → top-p)
  // ════════════════════════════════════════════════════════════
  onEnter("samp", (el) => {
    const VOCAB = [
      ["␣mat", 3.4], ["␣floor", 2.9], ["␣sofa", 2.5], ["␣bed", 2.2], ["␣roof", 1.7], ["␣table", 1.5],
      ["␣keyboard", 1.1], ["␣windowsill", 0.8], ["␣moon", -0.4], ["␣throne", -0.9], ["␣spaceship", -1.6], ["mat", -2.2],
    ];
    const seen = new Set();
    const hits = new Map();
    const ctl = { t: $("#s-t", el), k: $("#s-k", el), p: $("#s-p", el), r: $("#s-r", el) };
    const bars = $(".samp-bars", el);
    bars.innerHTML = VOCAB.map(([tok]) => `<div class="sbar" data-tok="${tok}"><div class="tok" title="click: toggle 'already generated'">${tok}</div><div class="trk"><div class="fill"></div></div><div class="pct"></div><div class="hits"></div></div>`).join("");
    let probs = [];
    function compute() {
      const T = +ctl.t.value, K = +ctl.k.value, P = +ctl.p.value, RP = +ctl.r.value;
      $("output[for=s-t]", el).textContent = fmt(T, 2);
      $("output[for=s-k]", el).textContent = K >= VOCAB.length ? "off" : K;
      $("output[for=s-p]", el).textContent = P >= 1 ? "off" : fmt(P, 2);
      $("output[for=s-r]", el).textContent = fmt(RP, 2);
      let logits = VOCAB.map(([tok, l]) => (seen.has(tok) ? (l > 0 ? l / RP : l * RP) : l));
      logits = logits.map((l) => l / T);
      const order = logits.map((l, i) => i).sort((a, b) => logits[b] - logits[a]);
      const keep = new Set(order.slice(0, K));
      const mx = Math.max(...logits);
      let e = logits.map((l, i) => (keep.has(i) ? Math.exp(l - mx) : 0));
      let z = e.reduce((a, b) => a + b, 0);
      let p = e.map((x) => x / z);
      if (P < 1) {
        let cum = 0; const keep2 = new Set();
        for (const i of order) { if (p[i] === 0) continue; keep2.add(i); cum += p[i]; if (cum >= P) break; }
        p = p.map((x, i) => (keep2.has(i) ? x : 0));
        z = p.reduce((a, b) => a + b, 0); p = p.map((x) => x / z);
      }
      probs = p;
      const pmax = Math.max(...p);
      Array.from(bars.children).forEach((row, i) => {
        row.classList.toggle("cut", p[i] === 0);
        row.querySelector(".tok").classList.toggle("seen", seen.has(VOCAB[i][0]));
        row.querySelector(".fill").style.width = `${p[i] === 0 ? 1.5 : Math.max(1.5, (p[i] / pmax) * 100)}%`;
        row.querySelector(".pct").textContent = p[i] === 0 ? "cut" : `${fmt(p[i] * 100, 1)}%`;
        row.querySelector(".hits").textContent = hits.get(VOCAB[i][0]) || "";
      });
      const ent = -p.reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0);
      $(".samp-ent", el).textContent = `${fmt(ent, 2)} bits`;
      $(".samp-live", el).textContent = p.filter((x) => x > 0).length;
    }
    function sampleOnce() {
      let r = Math.random(), i = 0;
      for (; i < probs.length; i++) { r -= probs[i]; if (r <= 0) break; }
      i = Math.min(i, probs.length - 1);
      if (probs[i] === 0) i = probs.indexOf(Math.max(...probs));
      return i;
    }
    function sample(n) {
      let last = 0;
      for (let j = 0; j < n; j++) { last = sampleOnce(); hits.set(VOCAB[last][0], (hits.get(VOCAB[last][0]) || 0) + 1); }
      Array.from(bars.children).forEach((r, i) => r.classList.toggle("hit", i === last));
      $(".samp-out", el).innerHTML = `<span class="ctx">The cat sat on the</span> <span class="gen">${VOCAB[last][0].replace("␣", "")}</span>`;
      compute();
    }
    Object.values(ctl).forEach((c) => c.addEventListener("input", () => { hits.clear(); compute(); }));
    bars.addEventListener("click", (e) => {
      const row = e.target.closest(".sbar"); if (!row || !e.target.classList.contains("tok")) return;
      const t = row.dataset.tok; seen.has(t) ? seen.delete(t) : seen.add(t); hits.clear(); compute();
    });
    $(".s-one", el).addEventListener("click", () => sample(1));
    $(".s-many", el).addEventListener("click", () => { hits.clear(); sample(100); });
    const PRESETS = { greedy: [0.1, 1, 1, 1], ours: [0.8, 12, 0.9, 1.5], chaos: [2, 12, 1, 1], strict: [0.8, 12, 0.9, 2.0] };
    el.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const [t, k, p, r] = PRESETS[b.dataset.preset];
      ctl.t.value = t; ctl.k.value = k; ctl.p.value = p; ctl.r.value = r; hits.clear(); compute();
    }));
    compute();
  });

  // ════════════════════════════════════════════════════════════
  // 5. Prompt explorer: base vs base (rep 2.0) vs chat fine-tune
  // ════════════════════════════════════════════════════════════
  onEnter("px", (el) => {
    const data = window.PROMPTS;
    const rows = data.rows;
    const list = $(".px-list", el);
    const COLS = [
      { key: "base", title: "Base model", meta: "3–7 tok/param · rep 1.5" },
      { key: "base_rp2", title: "Base, stricter", meta: "3–7 tok/param · rep 2.0" },
      { key: "chat", title: "Chat fine-tune", meta: "8 tok/param + SFT", cls: "chat" },
    ];
    const HL = {
      role: { re: /(^|\n)(\s*(?:user|assistant|system)\s*)(?=\n)/g, rep: (m, a, b) => `${a}<mark class="hl-role">${b}</mark>`, label: "leaked turn", color: "#c2185b" },
      pg:   { re: /(\[(?:Pg|pg)[^\]]*\]|\[\d+[a-z]?\])/g, rep: (m) => `<mark class="hl-pg">${m}</mark>`, label: "book artefact", color: "#b25d00" },
      glue: { re: /([A-Za-z]{26,})/g, rep: (m) => `<mark class="hl-glue">${m}</mark>`, label: "glued words", color: "#4a1fe0" },
      caps: { re: /((?:\b[A-Z’'"]{2,}[\s,.!?:;—-]+){5,})/g, rep: (m) => `<mark class="hl-caps">${m}</mark>`, label: "ALL-CAPS drift", color: "#075985" },
      md:   { re: /(\*\*[^*\n]+\*\*|^\s*\d+\.\s|#\w+)/gm, rep: (m) => `<mark class="hl-md">${m}</mark>`, label: "chat formatting", color: "#166534" },
    };
    const on = new Set(Object.keys(HL));
    const leg = $(".px-legend", el);
    leg.innerHTML = Object.entries(HL).map(([k, h]) => `<button class="on" data-k="${k}" style="border-color:${h.color};color:${h.color}">${h.label}</button>`).join("");
    function annotate(txt) {
      let h = esc(txt);
      for (const k of Object.keys(HL)) if (on.has(k)) h = h.replace(HL[k].re, HL[k].rep);
      return h;
    }
    list.innerHTML = rows.map((r, i) => {
      const chips = COLS.filter((c) => r[c.key]).map((c) => ({ base: "BASE", base_rp2: "REP2", chat: "CHAT" })[c.key]).join(" · ");
      return `<button data-i="${i}">${esc(r.prompt)}<span class="chips">${chips}</span></button>`;
    }).join("");
    const colsEl = $(".px-cols", el);
    colsEl.innerHTML = COLS.map((c) => `<div class="px-col ${c.cls || ""}"><header><b>${c.title}</b><span>${c.meta}</span></header><div class="px-text" data-k="${c.key}"></div></div>`).join("");
    let curI = 0, timers = [];
    function render(i, typewriter) {
      curI = i; timers.forEach(clearInterval); timers = [];
      list.querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
      const r = rows[i];
      $(".px-prompt", el).textContent = `“${r.prompt}”`;
      COLS.forEach((c) => {
        const box = colsEl.querySelector(`[data-k=${c.key}]`);
        const txt = r[c.key];
        box.scrollTop = 0;
        if (!txt) { box.className = "px-text empty"; box.textContent = "Not asked in this run."; return; }
        box.className = "px-text";
        if (!typewriter) { box.innerHTML = annotate(txt); return; }
        let n = 0; box.classList.add("caret");
        const t = setInterval(() => {
          n = Math.min(txt.length, n + 7);
          box.innerHTML = esc(txt.slice(0, n));
          box.scrollTop = box.scrollHeight;
          if (n >= txt.length) { clearInterval(t); box.classList.remove("caret"); box.innerHTML = annotate(txt); }
        }, 22);
        timers.push(t);
      });
    }
    list.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) render(+b.dataset.i, true); });
    leg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      on.has(b.dataset.k) ? on.delete(b.dataset.k) : on.add(b.dataset.k);
      b.classList.toggle("on"); render(curI, false);
    });
    $(".px-replay", el).addEventListener("click", () => render(curI, true));
    const start = rows.findIndex((r) => r.prompt.startsWith("We didnt start the fire"));
    render(start < 0 ? 0 : start, false);
  });

  // ════════════════════════════════════════════════════════════
  // 6. 3D token embeddings (Plotly, data from outputs/embeddings_3d_pca.html)
  // ════════════════════════════════════════════════════════════
  onEnter("emb", (el, slide) => {
    const E = window.EMBEDDINGS;
    const plot = $("#emb-plot", el);
    const STYLE = {
      "word (with leading space)": ["#8f7bff", "whole words <code>Ġthe Ġof</code>"],
      "word piece": ["#FF4B9B", "word pieces <code>he in re</code>"],
      "digits": ["#FF9a3c", "digits <code>Ġ19 Ġ200</code>"],
      "punctuation": ["#40d4ff", "punctuation <code>). ,</code>"],
      "other": ["#FBDE40", "other"],
    };
    const traces = E.traces.map((t) => t.name ? {
      type: "scatter3d", mode: "markers", name: t.name, x: t.x, y: t.y, z: t.z, text: t.text,
      hovertemplate: "<b>%{text}</b><extra>" + t.name + "</extra>",
      marker: { size: t.name.startsWith("word") ? 3.2 : 5, color: STYLE[t.name][0], opacity: 0.85 },
    } : {
      type: "scatter3d", mode: "text", x: t.x, y: t.y, z: t.z, text: t.text, hoverinfo: "skip", showlegend: false,
      textfont: { size: 13, color: "rgba(255,255,255,.9)", family: "JetBrains Mono" },
    });
    const ax = { showbackground: false, gridcolor: "rgba(255,255,255,.08)", zerolinecolor: "rgba(255,255,255,.18)", color: "rgba(255,255,255,.45)", title: { text: "" }, showspikes: false };
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: { xaxis: { ...ax, title: { text: "PC1" } }, yaxis: { ...ax, title: { text: "PC2" } }, zaxis: { ...ax, title: { text: "PC3" } }, camera: { eye: { x: 1.05, y: 0.95, z: 0.55 } }, aspectmode: "cube" },
      font: { family: "Source Sans 3", color: "#fff" },
      hoverlabel: { bgcolor: "#1b1040", bordercolor: "#FF4B9B", font: { family: "JetBrains Mono", size: 18 } },
    };
    Plotly.newPlot(plot, traces, layout, { displayModeBar: false, responsive: true });
    const leg = $(".emb-legend", el);
    leg.innerHTML = E.traces.map((t, i) => t.name ? `<button data-i="${i}"><i style="background:${STYLE[t.name][0]}"></i>${STYLE[t.name][1]}<small>${t.x.length}</small></button>` : "").join("") +
      `<button data-i="labels"><i style="background:#fff"></i>text labels<small>${E.traces.find((t) => !t.name)?.x.length || 0}</small></button>` +
      `<button data-act="spin" class="off"><i style="background:var(--grad)"></i>auto-rotate</button>`;
    let spin = null, ang = Math.atan2(0.95, 1.05);
    leg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.act === "spin") {
        if (spin) { cancelAnimationFrame(spin); spin = null; b.classList.add("off"); return; }
        b.classList.remove("off");
        const step = () => { ang += 0.006; Plotly.relayout(plot, { "scene.camera.eye": { x: 1.42 * Math.cos(ang), y: 1.42 * Math.sin(ang), z: 0.55 } }); spin = requestAnimationFrame(step); };
        step(); return;
      }
      const idx = b.dataset.i === "labels" ? E.traces.findIndex((t) => !t.name) : +b.dataset.i;
      const vis = b.classList.toggle("off") ? "legendonly" : true;
      Plotly.restyle(plot, { visible: b.dataset.i === "labels" ? (vis === true) : vis }, [idx]);
    });
    slide.addEventListener("slide:leave", () => { if (spin) { cancelAnimationFrame(spin); spin = null; } });
    slide.addEventListener("slide:enter", () => Plotly.Plots.resize(plot));
  });

  // ════════════════════════════════════════════════════════════
  // 7. OOM replay animation (src/train.py: shrink seq cap 25% per retry, floor = min_seq_len)
  // ════════════════════════════════════════════════════════════
  onEnter("oom", (el) => {
    const rows = $(".oom", el);
    function run() {
      rows.innerHTML = "";
      let seq = 4096; const fits = 2400, floor = 256; let k = 0;
      const tick = () => {
        const ok = seq <= fits;
        rows.insertAdjacentHTML("beforeend", `<div class="oom-row"><span class="mono">try ${k + 1}</span><div class="bar ${ok ? "" : "fail"}" style="width:${(seq / 4096) * 100}%"></div><span class="res" style="color:${ok ? "#1a9d4b" : "#e0336f"}">${seq} ${ok ? "✓ step" : "✗ OOM"}</span></div>`);
        if (ok || seq <= floor) return;
        seq = Math.max(floor, Math.floor(seq * 0.75)); k++;
        setTimeout(tick, 700);
      };
      tick();
    }
    $(".oom-run", el).addEventListener("click", run);
    run();
  });

  // ════════════════════════════════════════════════════════════
  // 7b. Pipeline diagram: GitHub → Drive → Colab, stepped by the deck's fragments (.pstep)
  // ════════════════════════════════════════════════════════════
  onEnter("pipe", (el) => {
    // [id, path, label, label x, label y, dashed?]
    const EDGES = [
      ["clone", "M370 85 C 480 85, 500 205, 596 215", "git clone", 480, 120],
      ["hfws", "M185 432 V 520", "stream", 250, 482],
      ["wsdrive", "M370 650 C 760 700, 1150 690, 1300 474", "upload shards + manifest", 830, 690],
      ["copy", "M1290 170 C 1200 170, 1170 240, 1084 250", "copy to SSD", 1190, 160],
      ["ckpt", "M1084 340 C 1170 340, 1200 310, 1286 310", "checkpoint / 500 steps", 1190, 360],
      ["resume", "M1290 420 C 1190 430, 1170 470, 1084 470", "latest.pt", 1190, 470, true],
      ["serve", "M1084 530 C 1180 560, 1200 627, 1286 627", "Gradio link", 1180, 610],
    ];
    const STEPS = [
      { edges: ["clone"], nodes: ["gh", "colab"], t: "<b>Clone the code.</b> The notebook's first cell runs <code>git clone</code>; the committed tokenizer comes with it." },
      { edges: ["hfws", "wsdrive"], nodes: ["hf", "ws", "drive"], t: "<b>Tokenize once.</b> Stream from HuggingFace, tokenize on a CPU box, upload shards + manifest to Drive." },
      { edges: ["copy"], nodes: ["drive", "colab"], t: "<b>Mount Drive, copy shards to the VM's SSD.</b> The manifest and tokenizer hash are checked first." },
      { edges: [], nodes: ["colab"], t: "<b>Train on the GPU.</b> bf16, 14.8k tokens/s. The notebook only calls <code>train()</code> from <code>src/</code>." },
      { edges: ["ckpt"], nodes: ["colab", "drive"], t: "<b>Every 500 steps:</b> atomic checkpoint → <code>latest.pt</code> / <code>best.pt</code> → upload (≈18 GB for the 1.5B)." },
      { edges: ["clone", "resume"], nodes: ["gh", "colab", "drive"], t: "<b>The VM dies.</b> A new one clones, mounts, resumes from <code>latest.pt</code> on the exact next batch." },
      { edges: ["resume", "serve"], nodes: ["drive", "colab", "aud"], t: "<b>Serve.</b> <code>LLM-inference.ipynb</code> loads a checkpoint from Drive and shares a Gradio link: the chat you've been using since the start." },
    ];
    const g = $(".pipe-edges", el);
    const badgeOf = {};
    STEPS.forEach((st, i) => st.edges.forEach((e) => { if (!(e in badgeOf)) badgeOf[e] = i + 1; }));
    g.innerHTML = EDGES.map(([id, d, label, lx, ly, dash]) => `
      <g class="pedge" data-e="${id}">
        <path class="ln${dash ? " dash" : ""}" d="${d}" marker-end="url(#pa)"/>
        <text x="${lx}" y="${ly + 36}" text-anchor="middle">${label}</text>
        <g class="badge" data-s="${badgeOf[id]}"><circle cx="${lx}" cy="${ly}" r="20"/><text x="${lx}" y="${ly + 6}">${badgeOf[id]}</text></g>
      </g>`).join("");
    const markers = Array.from(el.querySelectorAll(".pstep"));
    const cap = $(".pipe-caption", el);
    function render() {
      const n = markers.filter((m) => m.classList.contains("shown")).length;
      el.classList.toggle("stepping", n > 0);
      const st = STEPS[n - 1];
      el.querySelectorAll(".pnode").forEach((nd) => nd.classList.toggle("on", !!st && st.nodes.includes(nd.dataset.n)));
      el.querySelectorAll(".pedge").forEach((e) => e.classList.toggle("on", !!st && st.edges.includes(e.dataset.e)));
      $(".pc-n", cap).textContent = n ? String(n) : "→";
      $(".pc-t", cap).innerHTML = st ? st.t : "Seven stages from an empty Colab VM to a model you can chat with. Press → to walk through them.";
    }
    const mo = new MutationObserver(render);
    markers.forEach((m) => mo.observe(m, { attributes: true, attributeFilter: ["class"] }));
    g.addEventListener("click", (e) => {
      const b = e.target.closest(".badge"); if (!b) return;
      const n = +b.dataset.s;
      markers.forEach((m, i) => m.classList.toggle("shown", i < n));
    });
    render();
  });


  // ════════════════════════════════════════════════════════════
  // 11. Tokens per parameter: stages of the 1.5B run, same prompt at each stage
  // ════════════════════════════════════════════════════════════
  const TPP_PROMPT = "We didnt start the fire, it was always burning since the world was turning";
  const TPP_STAGES = [
  {
    "x": 3,
    "k": "First result",
    "t": "~3 tokens / param",
    "meta": "≈ 4.5B tokens · FineWeb-Edu + Gutenberg",
    "a": "",
    "why": "Examples from the first checkpoint will be added here."
  },
  {
    "x": 5,
    "k": "Base checkpoints",
    "t": "3–7 tokens / param",
    "meta": "≈ 4.5–10.5B tokens · rep 1.5",
    "a": "toward Asia and Africa... No. The flames shook from roof over chimney over town again for fear somebody'd burn his pile stovepipe away inside. We thought ourselves lucky we hadna done somethink foolish today. What wad yu say? One thing sure, they could put out fires without bothering neighbors by going ahead alone. As soon as I got started we wanted nobody stirring till morning in time for supper.\" He paused significantly--as …",
    "why": "Fluent, period-perfect prose that continues a <b>document</b>: dialect spellings, no idea it was asked anything."
  },
  {
    "x": 8,
    "k": "Today's model",
    "t": "~8 tokens / param + chat SFT",
    "meta": "≈ 12B tokens · then 1.22M chat conversations",
    "a": "The sky became black again, another day left undone that never ended\nAnd as I stood up to speak my truth to the gods alone\nMy hand had stayed at home, it was just not worth taking risks\nWith only four years to live on earth, let's do no harm\nBut all my life long, I've lived so many lives to come\nI'd rather face death every step of the journey / then die back down & tell the tale / no matter how hard you try\nIt’s still true to …",
    "why": "Same prompt, now answered <b>as a turn</b>, with line breaks and a consistent mood. The chat fine-tune brings the format; the extra tokens bring fluency.",
    "now": true
  }
];
  onEnter("tpp", (el) => {
    const MAX = 14, pos = (x) => `${(x / MAX) * 100}%`;
    $(".tpp-ticks", el).innerHTML = [0, 2, 4, 6, 8, 10, 12].map((t) => `<span style="left:${pos(t)}">${t}</span>`).join("");
    $(".tpp-marks", el).innerHTML = TPP_STAGES.map((st, i) =>
      `<button class="tpp-mark${st.now ? " now" : ""}" data-i="${i}" style="left:${pos(st.x)}"><i></i><span>${st.t.split(" ")[0]}</span></button>`).join("");
    $(".tpp-track", el).style.setProperty("--fill", pos(8));
    const show = (i) => {
      const st = TPP_STAGES[i];
      el.querySelectorAll(".tpp-mark").forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
      $(".tpp-k", el).textContent = st.k; $(".tpp-t", el).textContent = st.t; $(".tpp-meta", el).textContent = st.meta;
      $(".tpp-q", el).textContent = st.a ? `You: “${TPP_PROMPT}”` : "";
      $(".tpp-a", el).textContent = st.a;
      $(".tpp-a", el).classList.toggle("empty", !st.a);
      $(".tpp-why", el).innerHTML = st.why;
    };
    $(".tpp-marks", el).addEventListener("click", (e) => { const b = e.target.closest(".tpp-mark"); if (b) show(+b.dataset.i); });
    show(2);
  });

  // Per-browser memory for the live demos (training start time, Gradio link). Storage may be unavailable.
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* not persisted */ } },
  };

  // ════════════════════════════════════════════════════════════
  // 8. At a glance: tiny ↔ large (src/config.py MODEL_CONFIGS)
  // ════════════════════════════════════════════════════════════
  const PRESETS = {
    tiny: { params: "35.3M", params_s: "35,265,024 · 43× smaller", layers: "6", layers_s: "d_model = 512", heads: "8 / 4",
      heads_s: "head_dim = 64 · GQA 2:1", ffn: "1536", ffn_s: "3 × d_model", ctx: "2048", tps: "live", tps_s: "watch it in Colab", ppl: "live", ppl_s: "trained on stage today" },
    large: { params: "1.51B", params_s: "1,508.5M · tied embeddings", layers: "32", layers_s: "d_model = 2048", heads: "32 / 8",
      heads_s: "head_dim = 64 · GQA 4:1", ffn: "5632", ffn_s: "≈ 8/3 × d_model", ctx: "4096", tps: "14.8k", tps_s: "bf16 · one Colab G4 GPU", ppl: "6.56", ppl_s: "val loss 1.881 @ step 24k" },
  };
  onEnter("glance", (el) => {
    const set = (name) => {
      const P = PRESETS[name];
      el.querySelectorAll("[data-k]").forEach((n) => { n.textContent = P[n.dataset.k]; });
      el.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.p === name));
      $(".g-preset", el).textContent = `preset “${name}”`;
    };
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => set(b.dataset.p)));
    set("large");
  });

  // ════════════════════════════════════════════════════════════
  // 9. Live demo 2: mark when tiny training started; check-in shows elapsed time + loss scale
  // ════════════════════════════════════════════════════════════
  const elapsed = () => {
    const t0 = +store.get("tinyTrainStart");
    if (!t0) return null;
    const m = Math.floor((Date.now() - t0) / 60000);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
  };
  onEnter("live-train", (el) => {
    const status = $(".lt-status", el);
    const show = () => { const e = elapsed(); status.textContent = e ? `running for ${e}` : "not started yet"; };
    $(".lt-start", el).addEventListener("click", () => { store.set("tinyTrainStart", String(Date.now())); show(); });
    show(); setInterval(show, 20000);
  });

  onEnter("checkin", (el, slide) => {
    const LOSS0 = Math.log(32000), BIG = 1.881;
    const input = $(".ci-loss", el), mark = $(".lmark.tiny", el);
    const upd = () => {
      $(".ci-elapsed", el).textContent = elapsed() || "–";
      const L = parseFloat(input.value);
      if (!(L > 0)) { mark.hidden = true; $(".ci-ppl", el).textContent = ""; return; }
      store.set("tinyLoss", String(L));
      mark.hidden = false;
      const x = Math.min(1, Math.max(0, (LOSS0 - L) / (LOSS0 - BIG)));
      mark.style.setProperty("--x", x);
      $(".ci-mark", el).textContent = L.toFixed(2);
      const ppl = Math.exp(L);
      $(".ci-ppl", el).textContent = `perplexity ≈ ${ppl < 100 ? ppl.toFixed(1) : Math.round(ppl).toLocaleString()} · like choosing among that many tokens`;
    };
    input.value = store.get("tinyLoss") || "";
    input.addEventListener("input", upd);
    slide.addEventListener("slide:enter", upd);
    upd();
  });

  // ════════════════════════════════════════════════════════════
  // 10. Live demo 1: Gradio share link → QR code (LLM-inference.ipynb, GRADIO_SHARE = True)
  // ════════════════════════════════════════════════════════════
  // Static QR codes: any element with data-qr="<url>" (e.g. the "Go deeper" slide).
  document.querySelectorAll("[data-qr]").forEach((box) => {
    const qr = qrcode(0, "M");
    qr.addData(box.dataset.qr); qr.make();
    box.innerHTML = qr.createSvgTag({ cellSize: 8, margin: 0, scalable: true });
  });

  // "Get everything" slide: one card per entry in links.js; an empty url shows "link coming soon".
  const linksEl = document.getElementById("links");
  if (linksEl) {
    linksEl.innerHTML = (window.DECK_LINKS || []).map((l, i) => `
      <div class="link-card">
        <span class="where">${esc(l.where || "")}</span>
        <h3>${esc(l.title)}</h3>
        <div class="what">${esc(l.what || "")}</div>
        <div class="qr${l.url ? "" : " soon"}" data-i="${i}">${l.url ? "" : "link coming soon"}</div>
        <div class="url">${l.url ? esc(l.url.replace(/^https?:\/\//, "")) : ""}</div>
      </div>`).join("");
    linksEl.querySelectorAll(".qr:not(.soon)").forEach((box) => {
      const qr = qrcode(0, "M");
      qr.addData(window.DECK_LINKS[+box.dataset.i].url); qr.make();
      box.innerHTML = qr.createSvgTag({ cellSize: 8, margin: 0, scalable: true });
    });
  }

  onEnter("live-chat", (el) => {
    const input = $(".qr-input", el), img = $(".qr-img", el), urlEl = $(".qr-url", el);
    function render(url) {
      if (!url) return;
      const qr = qrcode(0, "M");
      qr.addData(url); qr.make();
      img.innerHTML = qr.createSvgTag({ cellSize: 8, margin: 0, scalable: true });
      urlEl.textContent = url.replace(/^https?:\/\//, "");
      input.value = url;
      store.set("gradioUrl", url);
    }
    const fromQuery = new URLSearchParams(location.search).get("live");
    render(fromQuery || store.get("gradioUrl"));
    $(".qr-set", el).addEventListener("click", () => render(input.value.trim()));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") render(input.value.trim()); });
    el.querySelectorAll(".try-list button").forEach((b) => b.addEventListener("click", () => {
      navigator.clipboard?.writeText(b.textContent).catch(() => {});
      el.querySelectorAll(".try-list button").forEach((x) => x.classList.toggle("copied", x === b));
    }));
  });
})();
