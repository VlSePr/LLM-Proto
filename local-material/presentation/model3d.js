/* 3D model of the 1.5B ("large" preset): click any level for an explanation. Needs THREE (assets/vendor/three.min.js). */
(function () {
  const root = document.getElementById("m3d");
  if (!root || !window.THREE) return;
  const slide = root.closest(".slide");
  const T = window.THREE;
  const $ = (s) => root.querySelector(s);

  // ─── Model facts (src/config.py MODEL_CONFIGS["large"]) ───
  const N = 32, M = (x) => (x >= 1e9 ? `${(x / 1e9).toFixed(2)}B` : `${(x / 1e6).toFixed(x < 1e7 ? 2 : 1)}M`);
  const P = { emb: 32000 * 2048, q: 2048 * 2048, kv: 512 * 2048, o: 2048 * 2048, ffn1: 2048 * 5632, norm: 2048 };
  P.attn = P.q + 2 * P.kv + P.o; P.ffn = 3 * P.ffn1; P.block = P.attn + P.ffn + 2 * P.norm;
  // Activation mean / std per layer, read off outputs/activation_stats.png (approximate).
  const STD = [3.3, 4.7, 5.4, 5.7, 6.6, 7.1, 7.5, 8.2, 8.4, 8.5, 8.6, 8.6, 8.7, 8.8, 8.9, 9.0,
    9.2, 9.3, 9.3, 9.5, 9.8, 10.3, 11.4, 12.2, 13.6, 14.8, 16.0, 17.0, 17.9, 19.2, 21.9, 24.4];
  const MEAN = [-0.08, -0.11, -0.13, -0.14, -0.16, -0.18, -0.19, -0.18, -0.18, -0.19, -0.19, -0.19, -0.19, -0.19, -0.19, -0.19,
    -0.20, -0.20, -0.20, -0.20, -0.21, -0.22, -0.23, -0.26, -0.30, -0.31, -0.36, -0.37, -0.39, -0.35, -0.26, -0.32];

  const band = (i) => i === 0
    ? "Layer 0 is special. It reads raw embeddings, and in our weights it has an outlier <code>W_k</code> and the smallest <code>attn_norm</code> gain of the whole stack."
    : i <= 7 ? "Early layers. The stream is still small (std 4–8). This is typically where word pieces get stitched into words and local order and syntax are resolved."
    : i <= 21 ? "The middle plateau. Activation std barely moves (8 → 10). Probing studies on LLaMA-style models usually find phrases, entities and facts represented here."
    : "Late layers. The stream grows fast (11 → 24) and the FFN weights get larger; <code>ffn_norm</code> gains shrink to compensate. These layers shape the next-token distribution.";

  const INFO = {
    tokens: { k: "Input", t: "Tokens in", s: "“The cat sat on the” → 5 token ids", b: "Text is split by the 32k BPE tokenizer. <code>Ġ</code> marks a leading space, so <code>Ġcat</code> is one token. The model only ever sees integers." },
    emb: { k: "Bottom of the stack", t: "Token embedding", s: `32,000 × 2048 · ${M(P.emb)} params`, b: "A lookup table: each token id picks one 2048-number row. That row becomes the start of the <b>residual stream</b>. The same matrix is reused, <b>tied</b>, as the LM head at the top." },
    stream: { k: "Runs through all 32 blocks", t: "Residual stream", s: "2048 numbers per token · thickness = activation std", b: "The model's working memory. Every block <i>reads</i> it through a norm and <i>adds</i> its result back, never overwriting. Its width here is drawn to scale with the measured activation std: ~3 at layer 0, ~24 at layer 31." },
    norm: { k: "Inside a block", t: "RMSNorm", s: "2048 learned gains · no bias", b: "Pre-norm: scale the stream to unit RMS <i>before</i> each sub-layer, so attention and the FFN always see well-behaved inputs while the stream itself keeps growing. Computed in fp32." },
    q: { k: "Attention · 32 heads", t: "Query heads", s: `W_q 2048 × 2048 · ${M(P.q)}`, b: "Each of the 32 heads asks its own question of the context with a 64-dim query: “which earlier tokens matter to me?” Scores are <code>softmax(QKᵀ/√64)</code> with a causal mask, so no peeking ahead." },
    kv: { k: "Attention · 8 KV heads", t: "Key / value heads (GQA)", s: `W_k, W_v 512 × 2048 · ${M(P.kv)} each`, b: "Only 8 key/value heads, each <b>shared by 4</b> query heads (the lines). That is grouped-query attention: a 4× smaller KV cache (64 KiB per token for the whole model) at almost no quality cost." },
    rope: { k: "Attention · position", t: "RoPE", s: "θᵢ = 10000^(−2i/64) · 0 params", b: "Rotary embeddings rotate each pair of q/k dimensions by an angle proportional to position, so <code>q·k</code> depends only on the <b>distance</b> between tokens. No position table, nothing to learn." },
    wo: { k: "Attention · output", t: "Output projection", s: `W_o 2048 × 2048 · ${M(P.o)}`, b: "Concatenates the 32 heads' results and mixes them back into one 2048-vector, which is <b>added</b> to the residual stream." },
    gate: { k: "SwiGLU · 1 of 3", t: "Gate projection", s: `W_gate 2048 → 5632 · ${M(P.ffn1)}`, b: "Expands to 5632 dims and passes through SiLU. This decides, per hidden unit, <b>how much</b> of the signal gets through." },
    up: { k: "SwiGLU · 2 of 3", t: "Up projection", s: `W_up 2048 → 5632 · ${M(P.ffn1)}`, b: "A second expansion carrying the actual content. It is multiplied element-wise by the gate." },
    mul: { k: "SwiGLU · the gate", t: "SiLU(gate) ⊙ up", s: "element-wise product · 5632 dims", b: "The “GLU” in SwiGLU. A learned, input-dependent valve on every hidden unit. It is the main reason SwiGLU beats a plain ReLU MLP at equal parameter count." },
    down: { k: "SwiGLU · 3 of 3", t: "Down projection", s: `W_down 5632 → 2048 · ${M(P.ffn1)}`, b: "Compresses back to 2048 and adds the result to the stream. The FFN holds ~77% of a block's parameters; it's where most of the “knowledge” lives, and what MoE later splits into experts." },
    final: { k: "Top of the stack", t: "Final RMSNorm", s: "2048 gains", b: "One last normalisation of the stream before it is turned into scores over the vocabulary." },
    head: { k: "Top of the stack", t: "LM head (tied)", s: "2048 → 32,000 · 0 new params", b: "Multiplies the final vector by the <b>same</b> embedding matrix, transposed, to get a score (logit) for every token in the vocabulary. The dashed line is the weight tying." },
    out: { k: "Output", t: "Next-token distribution", s: "softmax over 32,000 logits", b: "The model's only output: a probability for every possible next token. Sampling (temperature, top-k, top-p) picks one, it's appended, and the whole stack runs again. That's generation." },
  };
  const blockInfo = (i) => ({
    k: `Block ${i} of 32`, t: `Transformer block ${i}`,
    s: `${M(P.block)} params · attention ${M(P.attn)} + SwiGLU ${M(P.ffn)}`,
    b: `${band(i)}<br><br>It's opened up above: click the heads, the RoPE ring or the SwiGLU matrices.`,
    stats: [["activation std", `≈ ${STD[i]}`], ["activation mean", `≈ ${MEAN[i]}`], ["layer", `${i + 1} / 32`]],
  });

  function showInfo(inf) {
    $(".mi-kicker").textContent = inf.k;
    $(".mi-title").textContent = inf.t;
    $(".mi-sub").textContent = inf.s;
    $(".mi-body").innerHTML = inf.b;
    $(".mi-stats").innerHTML = (inf.stats || []).map(([a, b]) => `<div><b>${b}</b>${a}</div>`).join("");
  }
  const DEFAULT = { k: "The whole model", t: "1.51B parameters, 32 blocks", s: "tokens → embedding → 32 × (attention + SwiGLU) → logits",
    b: "Drag to rotate, scroll to zoom. <b>Click any layer</b> for an explanation; clicking a block opens it up. Or send a token through.",
    stats: [["parameters", "1,508.5M"], ["blocks", "32"], ["d_model", "2048"]] };

  let scene, camera, renderer, raycaster, pickables = [], blocks = [], topGroup, detail, detailStream, tieLine, particle, bars = [];
  let sel = null, hovered = null, spin = false, flowT = -1, running = false, inited = false;
  const view = { theta: 0.75, phi: 1.3, r: 62, ty: 12.8 }, goal = { ...view };
  const BASE = (i) => 0.7 + i * 0.62, GAP = 7.6, TOP = BASE(N) + 0.2;

  // Text sprite on a dark pill so it reads against any colour; drawn on top of the geometry.
  function label(text, { size = 0.55, color = "#ffffff", weight = 600, bg = "rgba(16,9,38,.88)", border = "rgba(255,255,255,.3)" } = {}) {
    const lines = String(text).split("\n"), LH = 76, PX = 26, PY = 14;
    const c = document.createElement("canvas"), ctx = c.getContext("2d");
    const font = `${weight} 64px "Source Sans 3", "Segoe UI", sans-serif`;
    ctx.font = font;
    const w = Math.ceil(Math.max(...lines.map((l) => ctx.measureText(l).width))) + PX * 2, h = lines.length * LH + PY * 2;
    c.width = w; c.height = h;
    if (bg) {
      ctx.fillStyle = bg; ctx.strokeStyle = border; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.roundRect(1.5, 1.5, w - 3, h - 3, 26); ctx.fill(); ctx.stroke();
    }
    ctx.font = font; ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    lines.forEach((l, i) => ctx.fillText(l, w / 2, PY + LH * i + LH / 2 + 3));
    const tex = new T.CanvasTexture(c); tex.anisotropy = 4;
    const s = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
    s.renderOrder = 10;
    s.scale.set((size * w) / LH, (size * h) / LH, 1);
    return s;
  }
  function box(w, h, d, color, key, opts = {}) {
    const mat = new T.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: opts.glow ?? 0.1, roughness: 0.5, metalness: 0.05,
      transparent: true, opacity: opts.opacity ?? 0.9 });
    const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat);
    m.add(new T.LineSegments(new T.EdgesGeometry(m.geometry), new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22 })));
    m.userData = { key, baseGlow: mat.emissiveIntensity, ...opts.data };
    pickables.push(m);
    return m;
  }
  function streamSeg(h, r) {
    const m = new T.Mesh(new T.CylinderGeometry(1, 1, h, 24, 1, true),
      new T.MeshStandardMaterial({ color: 0x9fe8ff, emissive: 0x40c8ff, emissiveIntensity: 1.0, transparent: true, opacity: 0.85 }));
    m.scale.set(r, 1, r); m.userData = { key: "stream", baseGlow: 0.9 }; pickables.push(m);
    return m;
  }
  const rOf = (std) => 0.1 + std * 0.012;
  const mix = (a, b, t) => new T.Color(a).lerp(new T.Color(b), t);

  function build() {
    scene = new T.Scene();
    camera = new T.PerspectiveCamera(38, 1, 0.1, 400);
    renderer = new T.WebGLRenderer({ canvas: $("canvas"), antialias: true, alpha: true });
    raycaster = new T.Raycaster();
    scene.add(new T.AmbientLight(0xffffff, 0.45));
    const key = new T.DirectionalLight(0xffffff, 0.75); key.position.set(10, 30, 20); scene.add(key);
    const rim = new T.PointLight(0xff4b9b, 1.2, 80); rim.position.set(-14, 12, -10); scene.add(rim);
    const rim2 = new T.PointLight(0x6042ff, 1.2, 80); rim2.position.set(14, 4, 12); scene.add(rim2);

    // input tokens
    ["The", "␣cat", "␣sat", "␣on", "␣the"].forEach((t, i) => {
      const m = box(1.5, 0.45, 0.8, 0xfbae40, "tokens", { glow: 0.3 });
      m.position.set(-4 + i * 2, -2.6, 0); scene.add(m);
      const l = label(t.replace("␣", " "), { size: 0.7 }); l.position.set(-4 + i * 2, -1.95, 0); scene.add(l);
    });
    // embedding
    const emb = box(10, 0.8, 4.4, 0xff7701, "emb", { glow: 0.25 }); emb.position.y = -0.9; scene.add(emb);
    const el0 = label("Token embedding · 65.5M", { size: 0.95 }); el0.position.set(10.5, -0.9, 0); scene.add(el0);
    const s0 = streamSeg(1.6, rOf(STD[0])); s0.position.y = -0.1; scene.add(s0);

    // 32 blocks
    for (let i = 0; i < N; i++) {
      const g = new T.Group(), t = i / (N - 1);
      const n1 = box(6.6, 0.03, 3.6, 0x9d8bff, "block", { opacity: 0.45, data: { layer: i, part: "norm" } }); n1.position.y = 0;
      const at = box(7.4, 0.18, 3.6, mix(0x7b5cff, 0x3d00f3, t).getHex(), "block", { glow: 0.25, data: { layer: i, part: "attn" } }); at.position.y = 0.11;
      const n2 = box(6.6, 0.03, 3.6, 0x9d8bff, "block", { opacity: 0.45, data: { layer: i, part: "norm" } }); n2.position.y = 0.24;
      const ff = box(8.2, 0.2, 3.8, mix(0xff4b9b, 0xff7701, t).getHex(), "block", { data: { layer: i, part: "ffn" } }); ff.position.y = 0.37;
      const st = streamSeg(0.62, rOf(STD[i])); st.position.y = 0.31;
      g.add(n1, at, n2, ff, st);
      if (i % 4 === 0 || i === N - 1) { const l = label(`L${i}`, { size: 0.7, color: "#e4dcff", bg: "rgba(16,9,38,.7)", border: "rgba(201,188,255,.35)" }); l.position.set(-6, 0.2, 0); g.add(l); }
      g.position.y = BASE(i); g.userData = { i, y: BASE(i) };
      scene.add(g); blocks.push(g);
    }
    const lb = label("× 32 blocks · 45.1M each", { size: 0.95 }); lb.position.set(11, BASE(16), 0); scene.add(lb);
    const la = label("attention", { size: 0.6, color: "#b9a8ff" }); la.position.set(-6.4, BASE(0) - 0.7, 0); scene.add(la);
    const lf = label("SwiGLU", { size: 0.6, color: "#ffa6cc" }); lf.position.set(6.6, BASE(0) - 0.7, 0); scene.add(lf);

    // top: final norm, LM head, output distribution
    topGroup = new T.Group(); topGroup.userData = { y: TOP };
    const fn = box(6.6, 0.06, 3.6, 0x9d8bff, "final", { opacity: 0.7 }); fn.position.y = 0.2;
    const hd = box(10, 0.8, 4.4, 0xff7701, "head", { opacity: 0.55, glow: 0.3 }); hd.position.y = 1.1;
    const hl = label("LM head (tied) → 32k logits", { size: 0.95 }); hl.position.set(11, 1.1, 0);
    topGroup.add(fn, hd, hl);
    const VOC = [["mat", 3.4], ["floor", 2.9], ["sofa", 2.5], ["bed", 2.2], ["roof", 1.7], ["table", 1.5]];
    const ex = VOC.map(([, l]) => Math.exp(l / 0.8)), z = ex.reduce((a, b) => a + b, 0) * 1.12;
    VOC.forEach(([w], i) => {
      const p = ex[i] / z, h = 0.3 + p * 9, x = -5 + i * 2;
      const b = box(1.1, h, 1.1, i === 0 ? 0xff4b9b : 0x8f7bff, "out", { glow: 0.35 });
      b.position.set(x, 1.7 + h / 2, 0); b.userData.h = h; topGroup.add(b); bars.push(b);
      const l = label(`${w}\n${Math.round(p * 100)}%`, { size: 0.72, bg: i === 0 ? "rgba(214,24,110,.95)" : undefined });
      l.position.set(x, 1.7 + h + 1.25, 0); topGroup.add(l);
    });
    const ttl = label("next token after “The cat sat on the”", { size: 0.85 }); ttl.position.set(0, 1.7 + 0.3 + (ex[0] / z) * 9 + 3.3, 0); topGroup.add(ttl);
    topGroup.position.y = TOP; scene.add(topGroup);
    tieLine = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3()]),
      new T.LineDashedMaterial({ color: 0xffb36b, dashSize: 0.5, gapSize: 0.35 }));
    scene.add(tieLine);

    // exploded detail of one block (placed in the gap)
    detail = new T.Group();
    const dn = box(6.6, 0.04, 3.6, 0x9d8bff, "norm", { opacity: 0.6 }); dn.position.y = 0.3;
    detail.add(dn);
    const kvX = (g) => -4.2 + g * 1.2;
    for (let g = 0; g < 8; g++) {
      const kv = box(0.62, 0.62, 0.62, 0xff4b9b, "kv", { glow: 0.35 }); kv.position.set(kvX(g), 1.3, 0); detail.add(kv);
      for (let q = 0; q < 4; q++) {
        const x = kvX(g) - 0.39 + q * 0.26;
        const qm = box(0.2, 0.2, 0.2, 0x8f7bff, "q", { glow: 0.4 }); qm.position.set(x, 2.55, 0); detail.add(qm);
        const ln = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(x, 2.45, 0), new T.Vector3(kvX(g), 1.61, 0)]),
          new T.LineBasicMaterial({ color: 0xc9bcff, transparent: true, opacity: 0.6 }));
        detail.add(ln);
      }
    }
    const rope = new T.Mesh(new T.TorusGeometry(0.9, 0.07, 12, 48), new T.MeshStandardMaterial({ color: 0xfbae40, emissive: 0xfbae40, emissiveIntensity: 0.6 }));
    rope.rotation.x = Math.PI / 2; rope.position.set(5.8, 1.9, 0); rope.userData = { key: "rope", baseGlow: 0.6, spin: true }; pickables.push(rope); detail.add(rope);
    const wo = box(5.2, 0.06, 2.4, 0x6042ff, "wo", { opacity: 0.85 }); wo.position.y = 3.35; detail.add(wo);
    const dn2 = box(6.6, 0.04, 3.6, 0x9d8bff, "norm", { opacity: 0.6 }); dn2.position.y = 4.05; detail.add(dn2);
    const gate = box(3.8, 0.1, 2.2, 0xff4b9b, "gate"); gate.position.set(-2.2, 4.85, 0); detail.add(gate);
    const up = box(3.8, 0.1, 2.2, 0xf26b43, "up"); up.position.set(2.2, 4.85, 0); detail.add(up);
    const mul = new T.Mesh(new T.SphereGeometry(0.36, 24, 16), new T.MeshStandardMaterial({ color: 0xffffff, emissive: 0xff7701, emissiveIntensity: 0.7 }));
    mul.position.y = 5.75; mul.userData = { key: "mul", baseGlow: 0.7 }; pickables.push(mul); detail.add(mul);
    const down = box(3.8, 0.1, 2.2, 0xff7701, "down"); down.position.y = 6.6; detail.add(down);
    [["RMSNorm", 0.3, -7.2], ["8 KV heads", 1.3, -7.2], ["32 query heads", 2.55, -7.4], ["RoPE", 2.75, 5.8], ["W_o", 3.35, -7.2],
      ["RMSNorm", 4.05, -7.2], ["W_gate", 4.85, -7.2], ["W_up", 4.85, 5.9], ["⊙ SiLU", 5.75, 1.6], ["W_down", 6.6, -7.2]]
      .forEach(([t, y, x]) => { const l = label(t, { size: 0.62, color: "#ffffff" }); l.position.set(x, y, 0.2); detail.add(l); });
    detailStream = streamSeg(GAP, rOf(STD[0])); detailStream.position.y = GAP / 2 - 0.15; detail.add(detailStream);
    detail.scale.setScalar(0.001); detail.visible = false; scene.add(detail);

    // token-flow particle
    particle = new T.Mesh(new T.SphereGeometry(0.45, 24, 16), new T.MeshBasicMaterial({ color: 0xffffff }));
    const glow = new T.PointLight(0xffffff, 2.2, 8); particle.add(glow);
    particle.visible = false; scene.add(particle);
  }

  function resize() {
    const v = $(".m3d-view"), r = v.getBoundingClientRect();
    if (!r.width) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    renderer.setSize(Math.round(r.width * dpr), Math.round(r.height * dpr), false);
    camera.aspect = v.clientWidth / v.clientHeight; camera.updateProjectionMatrix();
  }

  function setSel(i) {
    sel = i;
    if (i === null) { goal.ty = 12.8; goal.r = 62; return; }
    detailStream.scale.set(rOf(STD[i]), 1, rOf(STD[i]));
    detail.visible = true;
    goal.ty = BASE(i) + 3.6; goal.r = 25;
  }

  function pick(ev) {
    const c = $("canvas"), r = c.getBoundingClientRect();
    const v = new T.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(v, camera);
    const hit = raycaster.intersectObjects(pickables.filter((m) => m.visible && isShown(m)), false)[0];
    return hit ? hit.object : null;
  }
  function isShown(o) { for (let p = o; p; p = p.parent) if (p === detail && !detail.visible) return false; return true; }

  function click(obj) {
    if (!obj) return;
    const u = obj.userData;
    if (u.key === "block") {
      if (sel === u.layer) { showInfo(INFO[u.part === "ffn" ? "down" : u.part === "attn" ? "q" : "norm"]); return; }
      setSel(u.layer); showInfo(blockInfo(u.layer)); return;
    }
    showInfo(INFO[u.key]);
    // Output and LM head sit at the very top: bring the camera up close so the labels are readable.
    if (u.key === "out" || u.key === "head" || u.key === "final") { setSel(null); goal.ty = TOP + 3.5; goal.r = 24; goal.phi = 1.4; }
  }

  // ─── Orbit controls (drag / wheel) ───
  let drag = null;
  function bindInput() {
    const c = $("canvas");
    c.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.moved += Math.abs(dx) + Math.abs(dy);
        goal.theta -= dx * 0.008; goal.phi = Math.min(2.9, Math.max(0.25, goal.phi - dy * 0.006));
        drag.x = e.clientX; drag.y = e.clientY; return;
      }
      const o = pick(e); if (o !== hovered) { hovered = o; c.style.cursor = o ? "pointer" : "grab"; }
    });
    c.addEventListener("pointerup", (e) => { if (drag && drag.moved < 6) click(pick(e)); drag = null; });
    c.addEventListener("wheel", (e) => { e.preventDefault(); goal.r = Math.min(80, Math.max(8, goal.r * (1 + Math.sign(e.deltaY) * 0.1))); }, { passive: false });
    root.querySelectorAll("[data-a]").forEach((b) => b.addEventListener("click", () => {
      const a = b.dataset.a;
      if (a === "flow") { setSel(null); flowT = 0; showInfo({ k: "Forward pass", t: "Sending a token through", s: "“The cat sat on the” → ?", b: "Watch the token climb the residual stream: embedding lookup, then 32 blocks each adding attention and SwiGLU updates, then the LM head scores all 32,000 tokens." }); }
      if (a === "reset") { setSel(null); Object.assign(goal, { theta: 0.75, phi: 1.3, r: 62, ty: 12.8 }); showInfo(DEFAULT); }
      if (a === "spin") { spin = !spin; b.classList.toggle("on", spin); }
    }));
  }

  const lerp = (a, b, k) => a + (b - a) * k;
  function frame() {
    if (!running) return;
    requestAnimationFrame(frame);
    if (spin && !drag) goal.theta += 0.004;
    for (const k of ["theta", "phi", "r", "ty"]) view[k] = lerp(view[k], goal[k], 0.1);
    camera.position.set(view.r * Math.sin(view.phi) * Math.cos(view.theta), view.ty + view.r * Math.cos(view.phi), view.r * Math.sin(view.phi) * Math.sin(view.theta));
    camera.lookAt(0, view.ty, 0);

    blocks.forEach((g) => { const ty = g.userData.y + (sel !== null && g.userData.i > sel ? GAP : 0); g.position.y = lerp(g.position.y, ty, 0.12); });
    topGroup.position.y = lerp(topGroup.position.y, TOP + (sel !== null ? GAP : 0), 0.12);
    if (sel !== null) detail.position.y = BASE(sel) + 0.55;
    const ds = lerp(detail.scale.x, sel !== null ? 1 : 0.001, 0.14); detail.scale.setScalar(ds);
    if (sel === null && ds < 0.01) detail.visible = false;
    detail.children.forEach((c) => { if (c.userData.spin) c.rotation.z += 0.02; });

    // tie line: embedding corner ↔ LM head corner
    const pts = tieLine.geometry.attributes.position;
    pts.setXYZ(0, -5, -0.9, 2.2); pts.setXYZ(1, -5, topGroup.position.y + 1.1, 2.2); pts.needsUpdate = true; tieLine.computeLineDistances();

    // hover / flow glow
    const now = performance.now();
    pickables.forEach((m) => {
      if (m.material.emissiveIntensity === undefined) return;
      let g = m.userData.baseGlow ?? 0.2;
      if (m === hovered) g += 0.6;
      if (m.userData.flash && now - m.userData.flash < 500) g += 1.2 * (1 - (now - m.userData.flash) / 500);
      if (sel !== null && m.userData.key === "block" && m.userData.layer === sel) g += 0.5;
      m.material.emissiveIntensity = g;
    });
    if (flowT >= 0) {
      flowT += 0.006;
      const y = lerp(-2.6, TOP + 1.8, Math.min(1, flowT));
      particle.visible = true; particle.position.set(0, y, 0);
      blocks.forEach((g) => { if (Math.abs(g.position.y + 0.2 - y) < 0.25) g.children.forEach((c) => { if (c.userData) c.userData.flash = now; }); });
      bars.forEach((b, i) => { const k = flowT < 1 ? 0.02 : 1; b.scale.y = lerp(b.scale.y, k, 0.08); b.position.y = 1.7 + (b.userData.h * b.scale.y) / 2; void i; });
      if (flowT >= 1.35) { flowT = -1; particle.visible = false; showInfo({ ...INFO.out, b: INFO.out.b + " Here it bets on <b>“ mat”</b>." }); }
    }
    renderer.render(scene, camera);
  }

  // Handle for rehearsal / debugging from the console: model3d.select(12), model3d.flow().
  window.model3d = { select: (i) => { setSel(i); showInfo(blockInfo(i)); }, info: (k) => showInfo(INFO[k]), flow: () => root.querySelector('[data-a=flow]').click() };

  function start() {
    if (!inited) { inited = true; build(); bindInput(); showInfo(DEFAULT); window.addEventListener("resize", resize); }
    resize(); if (!running) { running = true; frame(); }
  }
  function stop() { running = false; }
  const go = () => (document.fonts ? document.fonts.ready.then(start) : start());
  slide.addEventListener("slide:enter", go);
  slide.addEventListener("slide:leave", stop);
  if (slide.classList.contains("active")) go();
})();
