"""Synthetic grayscale video generators: a bright Gaussian blob moving over a
textured background, with an optional global camera pan layered on top.

Everything is deterministic given a seed, and every generator returns both
the video array and the ground-truth motion (blob position per frame, pan
offset per frame) so downstream experiments can quantitatively check what
each transform recovers.
"""
import numpy as np


def _make_background(world_h, world_w, seed=0):
    """A static, textured world background so that a camera pan is actually
    visible in pixel values (panning over a flat background would be
    invisible). Uses a low-frequency sinusoidal grid plus a touch of smooth
    noise, normalized to [0, 0.5] so the blob (intensity up to ~1.0) stands
    out clearly on top.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(world_h), np.arange(world_w), indexing="ij")
    grid = 0.5 + 0.5 * np.sin(2 * np.pi * xx / 17.0) * np.cos(2 * np.pi * yy / 23.0)

    noise = rng.standard_normal((world_h // 4 + 2, world_w // 4 + 2))
    noise = np.kron(noise, np.ones((4, 4)))[:world_h, :world_w]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    bg = 0.35 * grid + 0.15 * noise
    return bg.astype(np.float32)


# Cardinal directions used as classification labels: 0=left,1=right,2=up,3=down
DIRECTION_VECTORS = [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)]
DIRECTION_NAMES = ["left", "right", "up", "down"]


def _make_checkerboard(world_h, world_w, scale, contrast, seed):
    """Parametrized textured background (checkerboard-ish grid + smooth
    noise), normalized to roughly [0, 0.5] so a bright blob stands out.
    `scale` sets the grid period (px); `contrast` scales its amplitude."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(world_h), np.arange(world_w), indexing="ij")
    grid = 0.5 + 0.5 * np.sin(2 * np.pi * xx / scale) * np.cos(2 * np.pi * yy / scale)

    noise = rng.standard_normal((world_h // 4 + 2, world_w // 4 + 2))
    noise = np.kron(noise, np.ones((4, 4)))[:world_h, :world_w]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    bg = contrast * grid + 0.10 * noise
    return np.clip(bg, 0.0, 0.6).astype(np.float32)


def generate_labeled_clip(
    direction_idx,
    n_frames=12,
    h=56,
    w=56,
    blob_speed=2.0,
    blob_sigma=3.0,
    blob_intensity=0.7,
    bg_scale=12.0,
    bg_contrast=0.3,
    noise_std=0.03,
    pan_speed=0.0,
    pan_direction=(0.0, 1.0),
    center_jitter=6.0,
    rng=None,
):
    """One labeled clip for Stage 3.

    The blob moves in its labeled cardinal direction *in the viewport* (so
    the label is always visibly present on screen), at `blob_speed` px/frame.
    Independently, the textured background pans at `pan_speed` px/frame in
    `pan_direction` -- this is a pure nuisance (carries no label info) that
    slides background texture past every pixel, exactly the Stage 1b
    camera-motion confound. Small per-frame Gaussian pixel noise is added.

    Returns (clip (n_frames,h,w) float32 in [0,1], label int).
    """
    if rng is None:
        rng = np.random.default_rng()

    bdy, bdx = DIRECTION_VECTORS[direction_idx]
    pdy, pdx = pan_direction
    pnorm = np.hypot(pdy, pdx) or 1.0
    pdy, pdx = pdy / pnorm, pdx / pnorm

    # World big enough for the viewport plus the full pan excursion.
    max_pan = pan_speed * (n_frames - 1)
    pan_margin = int(np.ceil(max_pan)) + 2
    blob_margin = int(np.ceil(blob_sigma * 4)) + 2
    margin = pan_margin + blob_margin
    world_h, world_w = h + 2 * margin, w + 2 * margin

    bg = _make_checkerboard(world_h, world_w, bg_scale, bg_contrast,
                            seed=int(rng.integers(1 << 30)))

    # Blob viewport trajectory centered (+ jitter) so it stays in frame for
    # any window length regardless of speed.
    jy = float(rng.uniform(-center_jitter, center_jitter))
    jx = float(rng.uniform(-center_jitter, center_jitter))
    vy0 = h / 2.0 + jy - blob_speed * bdy * (n_frames - 1) / 2.0
    vx0 = w / 2.0 + jx - blob_speed * bdx * (n_frames - 1) / 2.0

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    clip = np.zeros((n_frames, h, w), dtype=np.float32)

    for t in range(n_frames):
        oy = margin + int(round(pan_speed * t * pdy))
        ox = margin + int(round(pan_speed * t * pdx))
        oy = int(np.clip(oy, 0, world_h - h))
        ox = int(np.clip(ox, 0, world_w - w))
        frame = bg[oy:oy + h, ox:ox + w].copy()

        cy = vy0 + blob_speed * bdy * t
        cx = vx0 + blob_speed * bdx * t
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        frame = frame + blob_intensity * np.exp(-d2 / (2.0 * blob_sigma ** 2))

        if noise_std > 0:
            frame = frame + rng.normal(0.0, noise_std, size=frame.shape)
        clip[t] = np.clip(frame, 0.0, 1.0)

    return clip, direction_idx


def generate_pan_over_real_background(
    background,
    n_frames=16,
    h=180,
    w=256,
    pan_speed=4.0,
    pan_direction=(0.0, 1.0),
    add_blob=True,
    blob_speed=3.0,
    blob_direction=(1.0, 0.0),
    blob_sigma=6.0,
    blob_intensity=0.7,
):
    """Pan a (h, w) viewport across a large static REAL grayscale background
    image at `pan_speed` px/frame (simulating camera motion over real
    texture), optionally compositing a synthetic Gaussian blob that moves
    locally at its own `blob_speed`/`blob_direction` inside the viewport.

    This gives a clip with BOTH global camera motion (the pan, over real
    lighting/texture/compression) AND local object motion (the blob), with
    exact ground truth for each -- so we can test whether FreqST separates
    the two on real texture.

    `background` : (Hbg, Wbg) float array in [0, 1], must be large enough to
    fit the viewport plus the full pan excursion.
    """
    background = np.asarray(background, dtype=np.float32)
    bg_h, bg_w = background.shape

    pdy, pdx = pan_direction
    pnorm = np.hypot(pdy, pdx) or 1.0
    pdy, pdx = pdy / pnorm, pdx / pnorm

    bdy, bdx = blob_direction
    bnorm = np.hypot(bdy, bdx) or 1.0
    bdy, bdx = bdy / bnorm, bdx / bnorm

    # Start the pan viewport so the whole excursion stays inside the image.
    max_pan = pan_speed * (n_frames - 1)
    start_oy = int((bg_h - h - abs(max_pan * pdy)) / 2)
    start_ox = int((bg_w - w - abs(max_pan * pdx)) / 2)
    start_oy = max(0, start_oy)
    start_ox = max(0, start_ox)

    video = np.zeros((n_frames, h, w), dtype=np.float32)
    pan_offset = np.zeros((n_frames, 2), dtype=np.float32)
    blob_pos_viewport = np.zeros((n_frames, 2), dtype=np.float32)

    # Blob starts centered in the viewport, moves at its own velocity.
    blob_start = (h / 2.0, w / 2.0)

    for t in range(n_frames):
        pan_y = pan_speed * t * pdy
        pan_x = pan_speed * t * pdx
        oy = start_oy + int(round(pan_y))
        ox = start_ox + int(round(pan_x))
        oy = np.clip(oy, 0, bg_h - h)
        ox = np.clip(ox, 0, bg_w - w)

        frame = background[oy:oy + h, ox:ox + w].copy()

        if add_blob:
            by = blob_start[0] + blob_speed * t * bdy
            bx = blob_start[1] + blob_speed * t * bdx
            frame = frame + _gaussian_blob(h, w, (by, bx), blob_sigma, blob_intensity)
            blob_pos_viewport[t] = (by, bx)

        video[t] = np.clip(frame, 0.0, 1.0)
        pan_offset[t] = (pan_y, pan_x)

    gt = {
        "pan_offset": pan_offset,
        "blob_pos_viewport": blob_pos_viewport,
        "pan_direction": (pdy, pdx),
        "blob_direction": (bdy, bdx),
        "pan_speed": pan_speed,
        "blob_speed": blob_speed if add_blob else 0.0,
        "has_blob": add_blob,
    }
    return video, gt


def _gaussian_blob(h, w, center, sigma, intensity):
    cy, cx = center
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return intensity * np.exp(-d2 / (2.0 * sigma ** 2))


def generate_moving_blob_video(
    n_frames=32,
    h=64,
    w=64,
    blob_speed=3.0,
    blob_direction=(0.0, 1.0),
    blob_sigma=4.0,
    blob_intensity=0.8,
    pan_speed=0.0,
    pan_direction=(0.0, 1.0),
    start_blob_pos=None,
    seed=0,
):
    """Generate a synthetic (n_frames, h, w) grayscale video of a Gaussian
    blob moving at `blob_speed` px/frame along `blob_direction`, optionally
    with the whole camera panning at `pan_speed` px/frame along
    `pan_direction`.

    Directions are (dy, dx) unit-ish vectors (need not be normalized; speed
    scales them).

    Returns
    -------
    video : (n_frames, h, w) float32 array in [0, 1]
    gt : dict with
        'blob_pos_viewport': (n_frames, 2) array of (y, x) blob centers as
            seen in the (panned) viewport -- i.e. what an observer at the
            camera actually sees.
        'pan_offset': (n_frames, 2) array of (dy, dx) cumulative camera pan
            relative to frame 0.
        'blob_pos_world': (n_frames, 2) array of (y, x) blob centers in the
            static world frame (unaffected by camera pan).
    """
    bdy, bdx = blob_direction
    bnorm = np.hypot(bdy, bdx) or 1.0
    bdy, bdx = bdy / bnorm, bdx / bnorm

    pdy, pdx = pan_direction
    pnorm = np.hypot(pdy, pdx) or 1.0
    pdy, pdx = pdy / pnorm, pdx / pnorm

    max_pan = pan_speed * (n_frames - 1)
    margin = int(np.ceil(abs(max_pan))) + int(np.ceil(blob_sigma * 4)) + 4
    world_h, world_w = h + 2 * margin, w + 2 * margin

    background = _make_background(world_h, world_w, seed=seed)

    if start_blob_pos is None:
        start_blob_pos = (world_h / 2.0, world_w / 2.0)

    video = np.zeros((n_frames, h, w), dtype=np.float32)
    blob_pos_viewport = np.zeros((n_frames, 2), dtype=np.float32)
    pan_offset = np.zeros((n_frames, 2), dtype=np.float32)
    blob_pos_world = np.zeros((n_frames, 2), dtype=np.float32)

    for t in range(n_frames):
        pan_y = pan_speed * t * pdy
        pan_x = pan_speed * t * pdx
        pan_oy = margin + int(round(pan_y))
        pan_ox = margin + int(round(pan_x))

        crop = background[pan_oy:pan_oy + h, pan_ox:pan_ox + w]

        blob_wy = start_blob_pos[0] + blob_speed * t * bdy
        blob_wx = start_blob_pos[1] + blob_speed * t * bdx

        blob_vy = blob_wy - pan_oy
        blob_vx = blob_wx - pan_ox

        blob = _gaussian_blob(h, w, (blob_vy, blob_vx), blob_sigma, blob_intensity)
        frame = np.clip(crop + blob, 0.0, 1.0)

        video[t] = frame
        blob_pos_viewport[t] = (blob_vy, blob_vx)
        pan_offset[t] = (pan_y, pan_x)
        blob_pos_world[t] = (blob_wy, blob_wx)

    gt = {
        "blob_pos_viewport": blob_pos_viewport,
        "pan_offset": pan_offset,
        "blob_pos_world": blob_pos_world,
        "blob_direction": (bdy, bdx),
        "pan_direction": (pdy, pdx),
        "blob_speed": blob_speed,
        "pan_speed": pan_speed,
    }
    return video, gt
