"""Synthetic training data from the colon digital twin (Isaac Sim 6.2): RTX frames + YOLO labels.

Held-out split: TRAINING polyps are 0, 1, 2 (placed at random spots); the autopilot is tested on 3, 4, 5, which never
appear here. Camera poses: random points in the lumen; ~55% aimed at a nearby polyp from 1-6 cm, the rest looking down
the lumen or at bare wall (negatives, so folds and highlights stop looking like polyps).
Labels: polyp mesh vertices projected with the camera intrinsics (+ in-front / in-frame / distance checks).
Usage: ./run_isaac.sh synth_data.py <n_frames> <seed>   -> out/synth/images/*.jpg, out/synth/labels/*.txt
"""
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 512})

sys.path.insert(0, "/work")
import os

import numpy as np
import omni.timeline
import omni.usd
from PIL import Image

import colon_geom as cg
import scene

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rng = np.random.default_rng(SEED)
OUT = "/work/out/synth"
os.makedirs(f"{OUT}/images", exist_ok=True); os.makedirs(f"{OUT}/labels", exist_ok=True)
W, H = scene.W, scene.H
HFOV = 2 * np.arctan(20.955 / 2 / 7.3)
FX = (W / 2) / np.tan(HFOV / 2)

# 9 training lesions: polyps 0, 1, 2, three random spots each
spots = [(idx, float(rng.uniform(0.04, 0.37)), float(rng.uniform(0, 2 * np.pi))) for idx in (0, 1, 2) for _ in range(3)]
stage = omni.usd.get_context().get_stage()
cam, sensor, lesions = scene.build(stage, lesions=spots, depth=True)
verts = []
for idx, sp, tp in spots:                                          # vertices of each polyp mesh, for labels
    v, _, f, top, inward, _ = cg.polyp("/assets/polyp3d", idx, s_p=sp, th_p=tp)
    verts.append((v[np.unique(f)], top, inward))
omni.timeline.get_timeline_interface().play()
S = np.linspace(0, cg.L, 4000); CP, CT, CN, CB = cg.frames(S)


def project(pts, pos, fwd, up):
    right = np.cross(fwd, up); right /= np.linalg.norm(right); u = np.cross(right, fwd)
    d = pts - pos
    z = d @ fwd
    x = d @ right; y = d @ u
    ok = z > 0.003
    return np.stack([W / 2 + FX * x / np.maximum(z, 1e-6), H / 2 - FX * y / np.maximum(z, 1e-6)], 1), ok, z


first = True
for i in range(N):
    k = int(rng.integers(40, len(S) - 40))
    off = rng.normal(0, 0.35 * cg.R0, 2)
    pos = CP[k] + off[0] * CN[k] + off[1] * CB[k]
    mode = rng.random()
    if mode < 0.55:                                                 # look at a polyp from 1-6 cm
        j = int(rng.integers(len(verts)))
        _, top, inward = verts[j]
        dist = rng.uniform(0.01, 0.06)
        tk = int(np.argmin(np.sum((CP - top) ** 2, 1)))
        back = -CT[tk] * dist * rng.uniform(0.3, 1.0) + np.array(inward) * dist * rng.uniform(0.3, 1.0)
        pos = top + back
        fwd = (top - pos) + rng.normal(0, 0.006, 3)
    elif mode < 0.8:                                                # down the lumen
        fwd = CT[k] + rng.normal(0, 0.25, 3)
    else:                                                           # at bare wall
        fwd = CN[k] * np.cos(rng.uniform(0, 6.28)) + CB[k] * np.sin(rng.uniform(0, 6.28)) + CT[k] * rng.uniform(0.2, 1.0)
    fwd = fwd / np.linalg.norm(fwd)
    up = CB[k] - np.dot(CB[k], fwd) * fwd; up /= np.linalg.norm(up)
    roll = rng.uniform(0, 2 * np.pi)                                # endoscopes rotate freely
    right = np.cross(fwd, up); up = np.cos(roll) * up + np.sin(roll) * right
    # keep the tip inside the lumen
    kk = int(np.argmin(np.sum((CP - pos) ** 2, 1))); o = pos - CP[kk]; o -= np.dot(o, CT[kk]) * CT[kk]
    if np.linalg.norm(o) > 0.7 * cg.R0:
        pos = pos - o * (1 - 0.7 * cg.R0 / np.linalg.norm(o))
        if mode < 0.55:                                             # still aim at the chosen polyp after the move
            fwd = (top - pos) + rng.normal(0, 0.004, 3); fwd /= np.linalg.norm(fwd)
            up = CB[k] - np.dot(CB[k], fwd) * fwd; up /= np.linalg.norm(up)
    img = scene.grab(app, cam, sensor, pos, scene.look_quat(fwd, up), settle=8, first=first)   # annotators lag a few updates behind a pose jump
    first = False
    dm = scene.depth_map(sensor)
    lines = []
    for v, top, _ in verts:
        uv, ok, z = project(v, pos, fwd, up)
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.sum() < 30 or dm is None:
            continue
        px = uv[inside].astype(int)
        vis = np.abs(dm[px[:, 1], px[:, 0]] - z[inside]) < 0.0015     # vertex depth == rendered depth -> not occluded
        if vis.sum() < 30 or vis.mean() < 0.25 or np.median(z[inside]) > 0.07:
            continue
        pts = uv[inside][vis]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            continue
        lines.append(f"0 {(cx0 + cx1) / 2 / W:.5f} {(cy0 + cy1) / 2 / H:.5f} {(cx1 - cx0) / W:.5f} {(cy1 - cy0) / H:.5f}")
    name = f"syn{SEED}_{i:05d}"
    Image.fromarray(img).save(f"{OUT}/images/{name}.jpg", quality=92)
    open(f"{OUT}/labels/{name}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
    if i % 100 == 0:
        print("SYN", i, len(lines), flush=True)
print("SYNTH DONE", N, flush=True)
app.close()
