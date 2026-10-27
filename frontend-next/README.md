# compresskit site

A showcase of the toolkit's recorded results. Every number is read from `src/data/*.json`, which `tools/export_site_data.py` writes from real runs (VGG-16 is typed in from a Kaggle log and labelled `transcribed`).

```
npm install
npm run dev        # http://localhost:3000
npm run build && npm test       # unit tests incl. the number audit over the built HTML
npx playwright test             # needs a server on :3100 (npm run start -- -p 3100)
```

## How it is put together

- **One motion language** (`src/motion/`): the easing and durations in `tokens.ts`, `Scrolly` (the only scroll-pinning primitive), `MotionNumber`/`CountUp` and `Band` (a tone band); the scroll-drawn snake (`src/landing/snake`). Light/dark cross-fades come from the tone engine in `src/landing/tone/`, driven by `data-tone-band` sections, on every page.
- **The animation is the data** (`src/viz/`): `ModelStrip` draws a model's layers as columns of channels from a run's recorded prune stats; `AnimSawtooth` draws `prune_history`; `AnimBar` fills bars.
- **Interactive, still honest** (`src/explore/`, `src/lib/`): the explorer scrubs recorded stages and never interpolates results. The temperature, bit-width and budget controls recompute from recorded logits, weights and measurements with pure functions that are unit-tested against the recorded Python values (`derive.test.ts`, `budget.test.ts`).
- **Rules** (see `tests/`): no figure without a source (`dataAudit.test.ts`), evergreen palette only (`guards.spec.ts`), tone tokens instead of literal colours (`landingTone.test.ts`).

## Refreshing the data

```
./.venv/bin/python tools/export_site_data.py --only models,verify
./.venv/bin/python tools/export_site_data.py --vgg-json path/to/results.json --only models   # replace the transcribed VGG-16
```
