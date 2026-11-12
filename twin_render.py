"""M1: build the colon digital twin in Isaac Sim 6.2 and render endoscope stills -> out/colon.usd, out/still_*.png.
Usage (PC): ./run_isaac.sh twin_render.py [polyp_idx] [light_intensity]"""
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 512})

sys.path.insert(0, "/work")
import numpy as np
import omni.timeline
import omni.usd
from PIL import Image

import colon_geom as cg
import scene

stage = omni.usd.get_context().get_stage()
pidx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cam, sensor, meta, top, inward = scene.build(stage, pidx, float(sys.argv[2]) if len(sys.argv) > 2 else 3000.0)
stage.GetRootLayer().Export("/work/out/colon.usd")
omni.timeline.get_timeline_interface().play()

s = np.linspace(0, cg.L, cg.NS)
p, t, n, b = cg.frames(s)
k = int(0.30 / cg.L * (cg.NS - 1))
look = lambda a, c: c - a
views = {
    "start": (p[20], t[20], b[20]),
    "folds": (p[300], t[300], b[300]),
    "polyp_3cm": (p[k - 48], look(p[k - 48], top), b[k - 48]),
    "polyp_1cm": (top - 0.010 * inward * -1 + 0 * top, -inward, t[k]),
}
first = True
for name, (pos, fwd, up) in views.items():
    img = scene.grab(app, cam, sensor, pos, scene.look_quat(fwd, up), first=first)
    first = False
    Image.fromarray(img).save(f"/work/out/still_{name}.png")
    print("STILL", name, "mean", round(float(img.mean()), 1), flush=True)
print("POLYP", meta["id"], meta["size_mm"], "mm", "top", top.round(4).tolist(), flush=True)
app.close()
