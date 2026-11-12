"""Procedural colon digital twin (pure numpy, shared by Isaac Sim and the Three.js video).

Tube along a bending 3D centerline with crescent-shaped haustral folds; one real polyp (Depth-Anything heightfield +
SAM 2 mask + photo from the "Polyps in 3D" project) set into the wall. Units: metres. Z = insertion direction.
"""
import json
from pathlib import Path

import numpy as np

L, R0 = 0.40, 0.024                  # 40 cm of colon, ~4.8 cm lumen
NS, NT = 640, 112                    # samples along / around
FOLD_EVERY, FOLD_DEPTH, FOLD_W = 0.034, 0.30, 0.0035


def centerline(s):
    """Point on the centerline at arc parameter s (gentle S-bends, so a straight push would hit the wall)."""
    x = 0.045 * np.sin(2 * np.pi * s / L * 0.85)
    y = 0.030 * np.sin(2 * np.pi * s / L * 0.55 + 1.1)
    return np.stack([x, y, s], -1)


def frames(s):
    """Parallel-transport frames (T, N, B) along the centerline."""
    p = centerline(s)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.zeros_like(t)
    n0 = np.cross(t[0], [0, 1, 0]); n[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(s)):
        v = n[i - 1] - np.dot(n[i - 1], t[i]) * t[i]
        n[i] = v / np.linalg.norm(v)
    b = np.cross(t, n)
    return p, t, n, b


def radius(s, th, seed=7):
    rng = np.random.default_rng(seed)
    k = np.round(s / FOLD_EVERY)
    d = s - k * FOLD_EVERY
    ph = rng.uniform(0, 2 * np.pi, 64)[k.astype(int) % 64]
    crescent = 0.55 + 0.45 * np.cos(3 * th + ph)                # haustra are crescents, not full rings
    fold = FOLD_DEPTH * crescent * np.exp(-(d / FOLD_W) ** 2)
    wob = 0.06 * np.sin(2 * np.pi * s / 0.11 + 2 * th) + 0.04 * np.sin(5 * th + 7 * s / L)
    return R0 * (1 - fold + wob)


def tube(around=3, along=0.03):
    """Wall mesh. Normals are computed on the closed ring (no seam), then the seam column is duplicated for UVs.
    The mucosa texture repeats `around` times around the circumference and every `along` metres along the tube."""
    s = np.linspace(0, L, NS)
    th = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    p, t, n, b = frames(s)
    S, TH = np.meshgrid(s, th, indexing="ij")
    r = radius(S, TH)
    dirs = np.cos(TH)[..., None] * n[:, None] + np.sin(TH)[..., None] * b[:, None]
    pos = p[:, None] + r[..., None] * dirs                         # NS x NT x 3, closed ring
    i, j = np.meshgrid(np.arange(NS - 1), np.arange(NT), indexing="ij")
    a, bq, c, d = i * NT + j, i * NT + (j + 1) % NT, (i + 1) * NT + (j + 1) % NT, (i + 1) * NT + j
    closed = np.stack([np.stack([a, c, bq], -1), np.stack([a, d, c], -1)], -2).reshape(-1, 3)
    nrm = normals(pos.reshape(-1, 3), closed).reshape(NS, NT, 3)
    # open the ring at the seam for texture coordinates: duplicate column 0 as column NT
    pos2 = np.concatenate([pos, pos[:, :1]], 1); nrm2 = np.concatenate([nrm, nrm[:, :1]], 1)
    uv = np.stack([np.tile(np.arange(NT + 1) / NT * around, (NS, 1)), np.repeat((s / along)[:, None], NT + 1, 1)], -1)
    a, bq, c, d = i * (NT + 1) + j, i * (NT + 1) + j + 1, (i + 1) * (NT + 1) + j + 1, (i + 1) * (NT + 1) + j
    tris = np.stack([np.stack([a, c, bq], -1), np.stack([a, d, c], -1)], -2).reshape(-1, 3)
    return pos2.reshape(-1, 3), uv.reshape(-1, 2), tris, nrm2.reshape(-1, 3)


def polyp(assets, idx, s_p=0.30, th_p=1.9, sink=0.0004):
    """Real polyp heightfield as a mesh on the wall at (s_p, th_p). Returns verts, uvs, tris, centre, inward normal."""
    meta = json.loads((Path(assets) / "polyps.json").read_text())["polyps"][idx]
    g = 160
    e = np.frombuffer((Path(assets) / f"p{idx}.elev.u16").read_bytes(), "<u2").reshape(g, g) / 65535.0
    e0, e1 = meta["elev_range"]
    elev = e0 + e * (e1 - e0)
    mask = np.frombuffer((Path(assets) / f"p{idx}.mask.u8").read_bytes(), np.uint8).reshape(g, g) / 255.0
    mask = blur(mask, 3.0)                                         # soft, hole-free footprint
    h = blur(np.clip(elev - np.median(elev[mask < 0.1]), 0, None), 2.0) * mask
    h = h / max(h.max(), 1e-6)                                     # 0..1 relative height on the polyp
    size = meta["size_mm"] / 1000.0
    frac = max(np.sqrt((mask > 0.5).mean()), 0.2)                  # polyp share of the crop side
    side = size / frac                                             # physical side of the crop patch
    height = 0.55 * size                                           # sessile polyp: about half as tall as wide
    s = np.array([s_p - 0.002, s_p, s_p + 0.002])
    p, t, n, b = frames(s)
    c, tt = p[1], t[1]
    rad = np.cos(th_p) * n[1] + np.sin(th_p) * b[1]                # outward radial direction
    wall = c + radius(np.array(s_p), np.array(th_p)) * rad
    tang = np.cross(tt, rad)                                       # along the circumference
    u = (np.arange(g) / (g - 1) - 0.5) * side
    U, V = np.meshgrid(u, u, indexing="xy")
    base = wall[None, None] + U[..., None] * tang + V[..., None] * tt
    base = base + sink * rad                                       # sink the rim slightly into the wall (no z-fight)
    verts = base - (h * height)[..., None] * rad                   # grow inward, toward the lumen
    uv = np.stack(np.meshgrid(np.arange(g) / (g - 1), 1 - np.arange(g) / (g - 1), indexing="xy"), -1)
    keep = mask > 0.04
    i, j = np.meshgrid(np.arange(g - 1), np.arange(g - 1), indexing="ij")
    q = keep[i, j] & keep[i + 1, j] & keep[i, j + 1] & keep[i + 1, j + 1]
    a, bq, cc, d = i * g + j, i * g + j + 1, (i + 1) * g + j + 1, (i + 1) * g + j
    tris = np.stack([np.stack([a, bq, cc], -1), np.stack([a, cc, d], -1)], -2)[q].reshape(-1, 3)
    top = wall - height * rad
    return verts.reshape(-1, 3), uv.reshape(-1, 2), tris, top, -rad, meta


def blur(a, sigma):
    """Separable Gaussian blur (numpy only)."""
    r = int(3 * sigma); x = np.arange(-r, r + 1); k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, a)


def outer_bend_spots(n=3, lo=0.08, hi=0.36, min_gap=0.07):
    """(s, theta) of the outer wall at the n sharpest bends: where a forward-looking scope naturally faces the wall."""
    s = np.linspace(0, L, 2000)
    p, t, nn, b = frames(s)
    k = np.gradient(t, axis=0) / (s[1] - s[0])                     # curvature vector, points to the bend centre
    mag = np.linalg.norm(k, axis=1)
    order = np.argsort(-mag)
    picks = []
    for i in order:
        if lo <= s[i] <= hi and all(abs(s[i] - s[j]) >= min_gap for j in picks):
            picks.append(i)
        if len(picks) == n:
            break
    out = []
    for i in sorted(picks):
        o = -k[i] / mag[i]                                         # outer wall direction
        out.append((float(s[i]), float(np.arctan2(np.dot(o, b[i]), np.dot(o, nn[i]))), float(mag[i])))
    return out


def normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# lesions for the autopilot: AD 6 mm (p4), SSL 6 mm (p3), HP 9 mm (p5) on the outer wall of the 3 sharpest bends
LESIONS = [(idx, sp, th) for idx, (sp, th, _) in zip((4, 3, 5), outer_bend_spots(3))]


if __name__ == "__main__":
    import sys
    v, uv, f, _ = tube()
    print("tube", v.shape, f.shape, "radius range", radius(np.linspace(0, L, 500)[:, None], np.linspace(0, 6.28, 64)[None]).min(), R0)
    if len(sys.argv) > 1:
        pv, puv, pf, top, inward, meta = polyp(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
        print("polyp", meta["id"], meta["size_mm"], "mm", pv.shape, pf.shape, "top", top.round(4))
