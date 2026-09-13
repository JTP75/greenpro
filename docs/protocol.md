# Output contract: the 17×9 frame

This is the boundary between this repo (capture + processing) and whatever
sends frames to Will's Green Building display simulator/server (not built
yet — see [status.md](status.md)).

## In-process

`greenpro.pipeline.Pipeline.latest_grid()` returns the most recent frame as a
`(17, 9, 3)` `numpy.uint8` array, or `None` before the first frame arrives.
Indexing is `grid[row, col]`, `row` in `[0, 17)` top-to-bottom, `col` in
`[0, 9)` left-to-right, each entry an `[r, g, b]` triple in `0..255`.

## Over HTTP

`GET /frame.json` on the preview server:

```json
{
  "w": 9,
  "h": 17,
  "rgb": [[r, g, b], [r, g, b], ...],   // exactly 153 entries
  "frame_no": 1234,
  "fps": 19.8
}
```

`rgb` is row-major: index `row * 9 + col` gives cell `(row, col)`. Row 0 is
the top of the display, column 0 is the left.

## Rate limit

**Never send frames to the display faster than 30 FPS.** This is a hardware
constraint on the Green Building array itself (confirmed with its
maintainer; also documented in `17x9-Tetris/README.md`: `Display.send()`
should be called at most once every 0.033s). This repo's pipeline already
paces itself below that ceiling — see `greenpro/pipeline.py`'s
`MIN_FRAME_PERIOD` — so polling `/frame.json` or `latest_grid()` at up to
30Hz is always safe. Anything that fans this out to the actual display
(e.g. calling `Display.send()`) must itself not exceed that rate even if it
polls this pipeline faster.

## Worked example: adapting to `17x9-Tetris`'s `Display`/`Frame`

`17x9-Tetris/utilities/display.py` defines the interface the real Green
Building display server (and its `DummyDisplay` pygame stand-in) expects.
This is the smallest adapter from this repo's output to that interface:

```python
from utilities.display import Color  # from the 17x9-Tetris repo

def send_grid(display, grid):
    """grid: (17, 9, 3) uint8, as returned by Pipeline.latest_grid()."""
    frame = display.makeframe()
    for row in range(17):
        for col in range(9):
            r, g, b = grid[row, col]
            frame[row, col] = Color(int(r), int(g), int(b))
    display.send(frame)   # caller must not invoke this faster than 30 Hz
```

Test this adapter against `DummyDisplay` (a working pygame sink) before any
network code exists, e.g.:

```python
from utilities.dummy import DummyDisplay
display = DummyDisplay()
while True:
    grid = pipeline.latest_grid()
    if grid is not None:
        send_grid(display, grid)
    time.sleep(1 / 20)  # stay under 30 Hz
```
