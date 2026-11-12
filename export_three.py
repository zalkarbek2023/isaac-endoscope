"""Light-weight geometry for the Three.js "x-ray" view in the explainer: wall (decimated), polyps, centerline,
and the robot tip path from a run. Writes video/public/twin.json (positions in cm, rounded).
Usage (host venv): python export_three.py out/<run>   (needs colon_geom.py + polyp3d assets)"""
import json
import sys
from pathlib import Path

import numpy as np

import colon_geom as cg

A = str(Path.home() / "medai-colon/outputs/polyp3d/web")
run = Path(sys.argv[1]) if len(sys.argv) > 1 else None
cg.NS, cg.NT = 240, 48                                            # decimated wall for the browser
v, uv, f, n = cg.tube()
r = lambda a: np.round(np.asarray(a) * 100, 3).tolist()          # metres -> cm
out = {"wall": {"v": r(v), "f": f.astype(int).tolist()}, "polyps": [], "center": r(cg.centerline(np.linspace(0, cg.L, 200)))}
for idx, sp, th in cg.LESIONS:
    pv, puv, pf, top, inward, meta = cg.polyp(A, idx, s_p=sp, th_p=th)
    keep = np.unique(pf); remap = -np.ones(len(pv), int); remap[keep] = np.arange(len(keep))
    sub = pf[::3]                                                   # every third triangle is plenty for a glow blob
    out["polyps"].append({"id": meta["id"], "class": meta["class"], "size_mm": meta["size_mm"], "top": r(top),
                          "v": r(pv[keep]), "f": remap[sub].tolist()})
if run and (run / "run.jsonl").exists():
    L = [json.loads(x) for x in open(run / "run.jsonl")]
    out["path"] = [{"t": x["t"], "p": r(x["pos"]), "s": x["state"][0], "c": x["conf"], "b": x["box"]} for x in L]
    out["summary"] = json.loads((run / "summary.json").read_text())
Path("video/public").mkdir(parents=True, exist_ok=True)
Path("video/public/twin.json").write_text(json.dumps(out, separators=(",", ":")))
print("twin.json", round(Path("video/public/twin.json").stat().st_size / 1e6, 2), "MB", "path" in out)
