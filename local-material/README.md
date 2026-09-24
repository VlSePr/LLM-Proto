# EPAM.AI Conference 2026 — "Building an LLM from scratch"

Materials for the conference talk about this repository: an interactive HTML deck plus the raw
outputs and transcripts it was built from.

```
local-material/
├── presentation/          # the deck (open index.html, see below)
│   ├── index.html         # all slides, with speaker notes
│   ├── deck.css / deck.js # styling and the slide engine (keys, overview, notes window)
│   ├── widgets.js         # interactive slides (architecture, RoPE, GQA, sampling, prompt explorer, …)
│   ├── model3d.js         # the 1.5B in 3D (Three.js)
│   ├── links.js           # URLs on the "Get everything" slide; edit this file to add links
│   └── assets/            # fonts, backgrounds, logos, charts, data, vendored JS libraries
├── outputs/               # charts from LLM_proto.ipynb (Part D/E) for the 1.5B run at ~24k steps
└── transcripts/           # chat logs recorded in rehearsal (LLM-inference.ipynb chat UI)
```

## Running the deck

Serve the folder and open it in Chrome or Edge:

```bash
python -m http.server 8765 --directory local-material/presentation
```

Then go to <http://localhost:8765>. Everything is vendored, so the deck works offline (except the live demos).
Opening `index.html` straight from disk also works, but some browsers block the bundled fonts on `file://`.

| Key | Action |
|-----|--------|
| `→` / `Space` | next step or slide |
| `←` | back |
| `O` | overview grid (click a slide to jump) |
| `S` | speaker-notes window with a timer |
| `P` | full speaker script, all notes in order (printable / save as PDF) |
| `F` | fullscreen |
| `?` | help |

Slides are addressable as `#/N`, e.g. `http://localhost:8765/#/9`.

## Speaker notes

Every slide's notes have the same parts: **⏱ timing** (six slides are marked *skip if short on time*), a plain-language
**narration** you can read almost as-is (built on one set of analogies: the guessing game, the 32-floor building with a
shared notebook, the classroom, clock hands), **key clues** (numbers and facts), **Do** (what to click) and
**If asked**. Press `S` for the live notes window or `P` for the whole script.

## Before the talk

1. **Live demo 1 (chat with the 1.5B).** Run `LLM-inference.ipynb` up to section 6 with `GRADIO_SHARE = True`,
   then paste the `*.gradio.live` link on slide 5, or open the deck as `http://localhost:8765/?live=<link>`.
   The link is remembered in that browser.
2. **Live demo 2 (train the tiny model).** Run `LLM_proto.ipynb` with `MODEL_SIZE = "tiny"`, `RESUME_FROM = ""`.
   Press *Mark training start* on the live-demo slide; the check-in slide in Part 4 shows the elapsed time and
   places the current loss on a scale.
3. **Links.** Fill the SharePoint URLs in `presentation/links.js`; empty cards show "link coming soon".

For rehearsal, `model3d.select(12)` and `model3d.flow()` in the browser console drive the 3D slide.

## Transcripts

All recorded with the chat UI at temperature 0.8, top-k 50, top-p 0.9, 256 new tokens.

| File | Model | Repetition penalty |
|------|-------|--------------------|
| `base-rep1.5.txt` | base model, FineWeb-Edu + Project Gutenberg, 3–7 tokens/param | 1.5 |
| `base-rep2.0.txt` | same base model | 2.0 |
| `chat-sft.txt` | ~8 tokens/param, then fine-tuned on the chat mix (one continuous session) | 1.5 |

The deck's prompt explorer reads a parsed copy of these (`presentation/assets/data/prompts.js`).

## Credits and licences

- The dataset credits are on slide 3; the talk was inspired by Sebastian Raschka's
  *Build a Large Language Model (From Scratch)* (Manning, 2024).
- **Conference branding** (EPAM / EPAM.AI Conference logos and background artwork in `presentation/assets/bg`
  and `presentation/assets/logo`) is © EPAM Systems and comes from the official conference template. It is
  **not** covered by this repository's MIT licence.
- Fonts: Bebas Neue, Source Sans 3, JetBrains Mono (SIL Open Font License).
- Libraries: Plotly.js, Three.js, qrcode-generator (MIT).
