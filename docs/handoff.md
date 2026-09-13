# Handoff: network path verification (2026-09-13)

Session goal: **can we stream *a* 17×9 grid to Will's Green Building
simulator?** Answer: **yes, confirmed live against the `hazel-toad`
instance.** This doc is a snapshot for picking this back up in a new
session — see [status.md](status.md) for the living project state and
[simulator.md](simulator.md) for the protocol reference; this file won't be
kept up to date after today.

## Where the work is

- **Branch:** `netstream`, in its own git worktree at
  `C:\Users\pacel\misc\sundai140\greenpro-netstream` (sibling to, not
  inside, the main checkout).
- **Commits** (2, on top of `origin/main` as of branch creation):
  - `77dcfcf` — carries over `docs/simulator.md` and the README/status.md
    edits from an earlier session's doc-research work (was uncommitted on
    `main` when this session started).
  - `bc6431b` — the actual network-verification work: `greenpro/patterns.py`,
    `greenpro/sink.py`, and wiring through `config.py`/`run.py`/`server.py`.
- Working tree is clean (`git status` — nothing uncommitted).
- **Not pushed to `origin`**, and not merged into `main`. Also: `main`'s
  `HEAD` has moved to `61db80e` since this branch was cut from `2fe8d75`
  (another session is actively working there) — this branch has **not**
  been rebased onto that, so expect to reconcile before merging.

## Why a separate worktree

The main repo directory (`C:\Users\pacel\misc\sundai140\greenpro`) and the
Raspberry Pi were both explicitly held by another session this whole time.
**Neither was touched** — no reads, no writes, no commands run there. All
work happened in the `netstream` worktree, which only shares the `.git`
object store (read-only from this branch's perspective) with the main
checkout. If you're continuing this work, keep respecting that boundary
unless you've confirmed the other session is done.

## What was built

Two new modules, following patterns already established in the existing
`pipeline.py`/`config.py`/`server.py`:

- **`greenpro/patterns.py`** — synthetic 17×9 grid generators (`blink`,
  `corners`, `bar`, `rainbow`, `solid`) that publish straight into the
  existing `LatestFrameHolder`, deliberately bypassing the segmenter/geometry
  chain (which exists to correct *camera* framing and would rotate/crop a
  hand-crafted pattern). `PatternPipeline` is a duck-typed sibling of
  `Pipeline` — same `.holder`/`.start()`/`.stop()`/`.latest_grid()` surface —
  selected via the existing `--source` convention: `--source pattern:blink`.

- **`greenpro/sink.py`** — `WebDisplaySink` (stdlib `urllib.request`, POSTs
  the nested 17×9×3 JSON body to `/api/i/<slug>/frame`) and `FrameSender`
  (a background thread that sends each new grid, live-reconfigurable via the
  existing `POST /config`, with capped exponential backoff on failure that
  never kills the thread or the pipeline it's attached to).

- **`config.py`**: new `SinkConfig` (`enabled`, `base_url`, `instance`,
  `fps`, `timeout`, `retry_backoff`) — `fps` clamped to the existing
  `MAX_DISPLAY_FPS` (30), a third independent enforcement point alongside
  `capture.fps` and the pipeline's own pacer.

- **`run.py`**: `--send` / `--instance` / `--sink-url` flags; builds
  `PatternPipeline` when `capture.source` starts with `pattern:`.

- **`server.py`**: new `/sink.json` route (sender stats); one extra line in
  the existing `/` stats poll.

## How to run it

```bash
cd C:\Users\pacel\misc\sundai140\greenpro-netstream
.venv\Scripts\python.exe run.py --source pattern:blink --send --instance hazel-toad -v
# open http://localhost:8000/  (or curl /sink.json, /frame.json, /config)
```

A local venv already exists at `.venv/` in this worktree (gitignored, not
committed) with `requirements.txt` installed — **don't use the ambient
`python`/`pip` on PATH**: on this machine `python` resolves to an unrelated
`hermes-agent` venv (Python 3.11), and the shared Python 3.13 site-packages
had a locked `cv2.pyd` (presumably from another process) when this session
tried it. If `.venv/` is ever missing:

```bash
"C:\Users\pacel\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Live control, no restart needed:

```bash
curl -X POST http://localhost:8000/config -d '{"sink":{"enabled":false}}'   # stop sending
curl -X POST http://localhost:8000/config -d '{"sink":{"instance":"hazel-toad"}}'
```

**The sender is not currently running against `hazel-toad`** — the test
process was stopped at the end of this session. Nothing is actively sending
frames to the live instance right now.

## What was verified (live, against `hazel-toad`)

Full results and reasoning in `docs/status.md`'s "Verified this session"
section. Short version: ran `pattern:blink` with `--send` for an extended
period (thousands of frames sent). Confirmed live, by watching
`https://sundai.willsarg.com/hazel-toad?view=river`:

1. **Happy path** — frames arrive; user visually confirmed red top-left /
   blue bottom-right blinking at ~1Hz, exactly matching `patterns.blink()`.
2. **Orientation** — the diagonal result above also rules out every possible
   axis-flip at once (the display is a non-square 17×9 rectangle, so a
   rotation mismatch is geometrically impossible — only flips are). Confirms
   `grid[0][0]` = top-left holds all the way through the simulator's render.
   This stood in for the originally-planned separate `corners` pattern test.
3. **Live toggle** — `POST /config {"sink":{"enabled":false}}` froze the
   send counter immediately; re-enabling resumed it.
4. **Rate clamp** — `POST /config {"sink":{"fps":120}}` came back clamped to
   `30`.
5. **Failure handling** — pointing `sink.instance` at a nonexistent slug
   produced repeated `400 {"error":"bad instance name"}` with visibly
   increasing backoff, while the source pipeline kept publishing frames at
   its own unaffected rate. Two *organic* transient failures also happened
   during the run (a TLS handshake timeout, a Cloudflare "error code: 1101")
   and both self-recovered with no intervention. Restoring the real instance
   name recovered sending automatically.
6. **Clean shutdown** — verified by code inspection, not directly observed:
   the harness's background-task stop mechanism force-kills rather than
   sending a real Ctrl-C/SIGINT, so `run.py`'s own `KeyboardInterrupt`/
   `finally` shutdown path never actually ran in this session. The logic
   mirrors the existing `Pipeline.stop()` pattern already used elsewhere
   (bounded `threading.Event`/`Condition` waits, ≤1-2s worst case), but
   worth a real Ctrl-C test in an interactive terminal before trusting it
   fully unattended.

## Two real bugs found and fixed during this run

1. **Cloudflare blocks the default `urllib` User-Agent.** The simulator
   sits behind Cloudflare, which 403s (`error code: 1010`) a plain
   `urllib.request` POST because of its default `Python-urllib/x.y` UA.
   `WebDisplaySink` now always sends an explicit `User-Agent` header. This
   would bite anyone hand-rolling a client with bare `urllib` — `requests`
   is unaffected (different default UA), which is why the original
   protocol-reverse-engineering session's illustrative snippet didn't hit
   it.
2. **Config staleness across backoff.** `FrameSender` read `sink` config
   once per loop iteration, including right before an up-to-10s backoff
   sleep — so a live fix applied *during* an active backoff didn't take
   effect until the *next* iteration, even though `/sink.json`'s `target`
   field had already updated a cycle earlier (found by literally doing this
   during the failure-injection test, watching it not recover as fast as
   expected). Fixed by re-reading config immediately after the backoff wait,
   right before constructing the request.

## What's not done / natural next steps

- **Not touched at all:** the Pi, the real camera, the segmenter/model
  decisions — all explicitly out of scope this session (Pi was held).
- **Physical camera mounting** is still an open decision (per `status.md`)
  — today's orientation result confirms the *simulator's* row/col convention,
  not which way the physical camera is rotated on its mount.
- This branch needs to be **reconciled with `main`** before merging — `main`
  moved ahead during this session (see "Where the work is" above). Diff
  `netstream` against current `origin/main`/`main` before opening a PR.
- Nothing's been pushed to `origin` yet.
- `docs/simulator.md`'s open questions list a few things worth confirming if
  the event password for the real API docs ever becomes available (exact
  validation messages, whether `/frame` has its own rate limit, instance
  expiry, the one unexplained "error code: 1101").
- The connectivity/config side of `--source pattern:...` is fully wired and
  tested; when the Pi is available again, `--source picam` is a drop-in
  replacement — no other code changes needed for the network half.
