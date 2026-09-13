"""Rotation/flip/fit geometry and the area-average downscale to the 17x9
display grid.

The same rotate+flip+fit transform is applied to whatever is being viewed
(raw preview, mask preview, or the coverage float array on its way to the
grid) so that what you see in /raw.mjpg is exactly what feeds the display,
just before the final resize.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import DISPLAY_COLS, DISPLAY_ROWS, GeometryConfig, ShadingConfig


def apply_orientation(img: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Apply rotation and flips only (no fit/crop). Works on 2D or 3D arrays."""
    if cfg.rotation:
        k = (cfg.rotation // 90) % 4
        img = np.rot90(img, k=-k)  # rot90's positive k is CCW; we want CW for "rotate right"
    if cfg.hflip:
        img = np.fliplr(img)
    if cfg.vflip:
        img = np.flipud(img)
    return np.ascontiguousarray(img)


def _crop_rect(src_w: int, src_h: int, cfg: GeometryConfig) -> tuple[int, int, int, int]:
    """Compute the (x, y, w, h) crop rect that gets the source, after an
    optional zoom, down to the display's aspect ratio (DISPLAY_COLS x
    DISPLAY_ROWS), anchored per cfg.crop_anchor. Returns coordinates in the
    *zoomed* source's pixel space (caller applies zoom first)."""
    target_aspect = DISPLAY_COLS / DISPLAY_ROWS  # width / height
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # source is wider than target -> crop width, keep full height
        new_w = int(round(src_h * target_aspect))
        new_h = src_h
    else:
        # source is taller than target -> crop height, keep full width
        new_w = src_w
        new_h = int(round(src_w / target_aspect))

    new_w = max(1, min(new_w, src_w))
    new_h = max(1, min(new_h, src_h))

    if cfg.crop_anchor == "start":
        x, y = 0, 0
    elif cfg.crop_anchor == "end":
        x, y = src_w - new_w, src_h - new_h
    else:  # center
        x, y = (src_w - new_w) // 2, (src_h - new_h) // 2

    return x, y, new_w, new_h


def fit_to_display_aspect(img: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Apply zoom + fit (crop/squash/pad) so the result has the display's
    aspect ratio (9:17), without resizing to the final 9x17 grid -- that
    happens once, in downscale_to_grid, using the highest-res version
    available so INTER_AREA has real data to average.
    """
    h, w = img.shape[:2]

    if cfg.zoom > 1.0:
        zh, zw = int(round(h / cfg.zoom)), int(round(w / cfg.zoom))
        zh, zw = max(1, zh), max(1, zw)
        y0 = (h - zh) // 2
        x0 = (w - zw) // 2
        img = img[y0 : y0 + zh, x0 : x0 + zw]
        h, w = img.shape[:2]

    if cfg.fit == "squash":
        return img  # aspect handled implicitly by downscale's target size

    if cfg.fit == "crop":
        x, y, cw, ch = _crop_rect(w, h, cfg)
        return img[y : y + ch, x : x + cw]

    # pad: letterbox to target aspect using zeros (black bars)
    target_aspect = DISPLAY_COLS / DISPLAY_ROWS
    src_aspect = w / h
    if src_aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        pad = new_h - h
        top, bottom = pad // 2, pad - pad // 2
        pad_spec = ((top, bottom), (0, 0)) if img.ndim == 2 else ((top, bottom), (0, 0), (0, 0))
    else:
        new_w = int(round(h * target_aspect))
        pad = new_w - w
        left, right = pad // 2, pad - pad // 2
        pad_spec = ((0, 0), (left, right)) if img.ndim == 2 else ((0, 0), (left, right), (0, 0))
    return np.pad(img, pad_spec, mode="constant", constant_values=0)


def crop_overlay_rect(src_w: int, src_h: int, cfg: GeometryConfig) -> tuple[int, int, int, int] | None:
    """For the /raw.mjpg preview: returns the crop rect (post-zoom, in
    oriented-source pixel space) to draw as an outline, or None if fit isn't
    "crop". Lets you see exactly what will and won't reach the display."""
    if cfg.fit != "crop":
        return None
    if cfg.zoom > 1.0:
        zh, zw = int(round(src_h / cfg.zoom)), int(round(src_w / cfg.zoom))
        y0, x0 = (src_h - zh) // 2, (src_w - zw) // 2
        x, y, cw, ch = _crop_rect(zw, zh, cfg)
        return x0 + x, y0 + y, cw, ch
    return _crop_rect(src_w, src_h, cfg)


def downscale_coverage(mask_f32: np.ndarray) -> np.ndarray:
    """Area-average a float32 mask (values in [0, 1], any HxW) down to the
    (DISPLAY_ROWS, DISPLAY_COLS) grid. cv2.resize with INTER_AREA computes
    exactly the per-cell area average requested -- each output cell's value
    is the mean of the input pixels it covers -- and works correctly even
    when the source dimensions aren't integer multiples of the grid.
    """
    resized = cv2.resize(
        mask_f32,
        (DISPLAY_COLS, DISPLAY_ROWS),  # cv2 takes (width, height)
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(resized, 0.0, 1.0)


def shade_grid(coverage: np.ndarray, cfg: ShadingConfig) -> np.ndarray:
    """Turn a (17, 9) coverage array in [0, 1] into a (17, 9, 3) uint8 RGB
    grid by interpolating between bg_color and fg_color."""
    c = coverage.copy()

    if cfg.min_coverage > 0:
        c[c < cfg.min_coverage] = 0.0

    if cfg.shading == "threshold":
        c = (c >= cfg.threshold).astype(np.float32)
    elif cfg.shading == "gamma":
        c = np.power(np.clip(c, 0.0, 1.0), cfg.gamma)

    if cfg.levels > 0:
        c = np.round(c * cfg.levels) / cfg.levels

    fg = np.array(cfg.fg_color, dtype=np.float32)
    bg = np.array(cfg.bg_color, dtype=np.float32)
    grid = bg[None, None, :] + (fg - bg)[None, None, :] * c[..., None]
    return np.clip(grid, 0, 255).astype(np.uint8)
