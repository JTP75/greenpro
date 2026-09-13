# Will's Green Building simulator

Notes on `https://sundai.willsarg.com`, the "Green Building sim (POC)" site
that stands in for the real MIT Green Building display while this project is
developed. This is the target for the network-delivery step called out as
out of scope in [status.md](status.md) and as step 4 in the main
[README](../README.md).

**How this was gathered:** the site's own "API docs" page is password-gated
(event password) and that password wasn't available in this session. Instead
this is reverse-engineered from the site's public pages, its unminified-enough
JS bundle (`/assets/view-*.js`), and live probing of the endpoints below with
`curl` on 2026-09-13, all against the user's own instance. Treat anything
not explicitly marked "confirmed by probing" as inferred and worth
double-checking once real docs access is available.

## Instances

The site is organized around named "instances," each an independent
simulated display:

- **Create**: `POST /api/instances` with JSON body `{"password": "..."}`
  (an event password, not the same thing as per-instance access). Returns
  `{"view_url": "..."}` on success; the front page redirects the browser
  there.
- **Instance slug**: adjective-noun, e.g. `curious-cat` (site's own
  placeholder text) or this project's `hazel-toad`. The slug is both the
  instance's identifier and, functionally, its access token — anyone who
  knows the slug can watch it and (per below) POST frames to it. There's no
  visible per-frame auth beyond knowing the slug.
- **Watch**: open `https://sundai.willsarg.com/<slug>` directly, or use the
  "Watch an instance" box on the front page (`/`), which just navigates to
  `/<slug>`. No password needed to watch.
- **Demos**: `GET /demos/demos.json` lists pre-built looping animations
  (`slug`, `title`, `description`, `fps`, `frames`) that play with no
  instance at all, at `/demo/<demo-slug>`. Confirmed by probing — see
  `demos.json` contents below. Not relevant to sending real frames, but
  useful for seeing the display format ahead of time.

## The view page (`/<slug>?view=...`)

This is what `https://sundai.willsarg.com/hazel-toad?view=river` is —
a Three.js-rendered 3D scene of the Green Building with the instance's
17×9 pixel grid mapped onto its windows.

- `?view=` — `close` (close-up), `street` (full building), or `river`
  (across the river; matches the real vantage point most people photograph
  the building from). Selectable live via the `<select id="view">` in the UI
  too.
- Checkboxes: "Real tree line" (hides the bottom two rows, to match what's
  actually visible from the street), "Display-first framing" (bigger tower,
  skyline squeezed in), "GPU effects" (bloom, water, haze, film grain —
  cosmetic only, doesn't affect the underlying 17×9 data).
- The left panel's status line reads `connecting…` → `live · waiting for
  frames` → `live` once frames arrive, or `demo · N frames @ F fps, looping`
  for demo playback. It also shows `reconnecting…` if the websocket drops.
- "hide/show connection info" toggles a panel with the exact `Send URL`,
  a ready-to-run `curl` command, and a Python snippet, all pre-filled with
  the current instance's slug (see next section).

## Sending frames: `POST /api/i/<slug>/frame`

**Confirmed by probing against the `hazel-toad` instance.**

```
POST https://sundai.willsarg.com/api/i/<slug>/frame
Content-Type: application/json

<body: a JSON array of 17 rows, each row an array of 9 [r,g,b] triples>
```

i.e. shape `(17, 9, 3)`, row-major, **row 0 is the top of the display,
column 0 is the left** — this lines up exactly with this repo's own
`Pipeline.latest_grid()` / `/frame.json` convention (see
[protocol.md](protocol.md)), just nested instead of flattened. Each `r`/`g`/`b`
is an integer `0..255`.

```bash
curl -X POST https://sundai.willsarg.com/api/i/hazel-toad/frame \
  -H 'Content-Type: application/json' \
  -d @frame.json
```

Responses observed:

- **Success**: `204 No Content`, empty body.
- **Wrong shape**: `400 Bad Request`, JSON body `{"error": "expected 17 rows"}`
  (and presumably analogous messages for wrong column count / malformed
  triples — not individually confirmed).
- CORS is wide open (`Access-Control-Allow-Origin: *`), so this endpoint can
  be called directly from a browser too, not just server-side.
- No auth header or token beyond the slug itself was required in testing.

### Adapting this repo's grid to the expected body

This repo's `(17, 9, 3)` `numpy.uint8` array from `Pipeline.latest_grid()`
maps directly:

```python
import json
import requests

def send_grid(slug, grid):
    """grid: (17, 9, 3) uint8 array, as returned by Pipeline.latest_grid()."""
    body = [[[int(v) for v in cell] for cell in row] for row in grid.tolist()]
    r = requests.post(
        f"https://sundai.willsarg.com/api/i/{slug}/frame",
        json=body,
        timeout=2,
    )
    r.raise_for_status()  # raises on anything but 204
```

Same **30 FPS ceiling** applies here as to the real hardware (see the
README's "Display rate limit" section) — this repo's pipeline already paces
itself under that, so calling `send_grid` once per `latest_grid()` update is
safe as-is; don't add a faster polling loop on top of it.

### `gbsim` — the site's own Python client

The view page's connection panel also shows a Python snippet using a
`gbsim` package, mirroring the `Display`/`Frame`/`Color` interface this
project's README references from `17x9-Tetris`:

```python
from gbsim import WebDisplay, Color

d = WebDisplay("hazel-toad", "https://sundai.willsarg.com/api")
f = d.makeframe()
f[0][0] = Color(255, 0, 0)
d.send(f)
```

`gbsim` is **not on PyPI** (checked — 404) — it's presumably distributed
alongside the `17x9-Tetris` repo or event materials rather than published
standalone. If it surfaces, using it directly would let this project's
future network-delivery adapter target the exact same `Display`/`Frame`
interface `protocol.md`'s worked example already assumes, with `WebDisplay`
just being the HTTP-backed implementation of the interface `DummyDisplay`
fakes locally. Until then, the raw `POST .../frame` call above is a
complete substitute.

## The view websocket (browser side, informational)

Not something this project needs to call, but useful for understanding how
frames flow once sent — and it's the same 17×9×3 wire layout in binary:

- The view page opens `wss://sundai.willsarg.com/api/i/<slug>/view` and
  receives binary messages (`binaryType = "arraybuffer"`).
- A **plain live frame** is exactly 459 bytes (= 17 × 9 × 3) of raw RGB,
  row-major, no header — presumably exactly what a `POST .../frame` body
  gets turned into server-side before fan-out.
- A **demo/clip message** starts with a one-byte magic `0x43` (`'C'`),
  followed by 1 byte FPS, 2 bytes little-endian frame count `n`, then
  `n × 459` bytes of concatenated frames — this is also the format of the
  static `/demos/<slug>.bin` files the demo pages fetch directly (confirmed
  the `demos.json` listing exists; the `.bin` format itself is inferred from
  the parsing code, not independently probed).
- On disconnect the page shows `reconnecting…` and retries after 1s.

## Open questions / not yet confirmed

- Exact validation messages for wrong column count or malformed `[r,g,b]`
  entries (only "expected 17 rows" was actually triggered).
- Whether `POST .../frame` has its own rate limit independent of the 30 FPS
  hardware ceiling, and what happens if it's exceeded.
- Contents of the password-gated `/docs` page — may formalize/supersede
  anything above. Worth revisiting if the event password becomes available.
- Whether instances expire, and after how long.
