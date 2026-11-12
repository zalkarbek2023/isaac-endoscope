"""M2 domain-gap check: render every one of the 6 real polyps in the twin at 1.5 and 3 cm -> out/ptest/p<i>_<d>.png.
Then run detector_server.py --test on them (host venv)."""
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

os.makedirs("/work/out/ptest", exist_ok=True)
stage = omni.usd.get_context().get_stage()
cam, sensor, meta, top, inward = scene.build(stage, int(sys.argv[1]))
omni.timeline.get_timeline_interface().play()
s = np.linspace(0, cg.L, cg.NS); p, t, n, b = cg.frames(s)
k = int(0.30 / cg.L * (cg.NS - 1))
first = True
for d in (0.015, 0.03):
    # oblique view like a real endoscope: from the lumen centre, d before the polyp along the tube, looking at it
    kk = max(0, k - int(d / cg.L * (cg.NS - 1)))
    pos = p[kk] + 0.3 * (top - p[kk]) * 0 + 0.0
    img = scene.grab(app, cam, sensor, pos, scene.look_quat(top - pos, b[kk]), first=first); first = False
    Image.fromarray(img).save(f"/work/out/ptest/p{sys.argv[1]}_{int(d * 1000)}mm.png")
    # face-on view from d away
    pos = top + d * inward
    img = scene.grab(app, cam, sensor, pos, scene.look_quat(-inward, t[k]))
    Image.fromarray(img).save(f"/work/out/ptest/p{sys.argv[1]}_{int(d * 1000)}mm_face.png")
print("PTEST", sys.argv[1], meta["id"], flush=True)
app.close()
