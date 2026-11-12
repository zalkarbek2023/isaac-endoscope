"""M3: autonomous robotic endoscope in the colon digital twin (Isaac Sim 6.2, headless).

Each step: the tip camera renders an RTX frame -> sent to the TensorRT polyp detector (detector_server.py, host,
127.0.0.1:5577) -> a finite-state controller steers the bending tip and the insertion:
  NAVIGATE  follow the lumen (darkest region of the frame, as in real endoscopy), advance 2 cm/s
  APPROACH  alarm = conf >= 0.45 on 5 frames in a row (colonoscopy-ai rule): steer to the box, slow down
  INSPECT   box big and centred -> hold 1.5 s, log the find; then a 2.5 s alarm cooldown and back to NAVIGATE
The robot is kinematic (insertion + 2-axis tip bending with a rate limit), not a soft-body simulation.
Outputs: out/<run>/frames/%05d.jpg, out/<run>/run.jsonl, out/<run>/summary.json, out/<run>/scene.json
Usage (PC): python detector_server.py &   then   ./run_isaac.sh endo_sim.py <run_name> [start_offset_m] [seed]
"""
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 512})

sys.path.insert(0, "/work")
import json
import os
import socket
import struct
import time

import numpy as np
import omni.timeline
import omni.usd
from PIL import Image, ImageFilter

import colon_geom as cg
import scene

RUN = sys.argv[1] if len(sys.argv) > 1 else "run0"
START_OFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUT = f"/work/out/{RUN}"
os.makedirs(f"{OUT}/frames", exist_ok=True)
W, H = scene.W, scene.H
FPS = 30.0
DT = 1.0 / FPS
HFOV = 2 * np.arctan(20.955 / 2 / 7.3)
VFOV = 2 * np.arctan(np.tan(HFOV / 2) * H / W)
V_NAV, V_APP = 0.015, 0.007                  # m/s
MAX_TURN = np.radians(70)                    # tip bending rate limit, rad/s
ALARM_CONF, ALARM_VOTES, ALARM_WINDOW = 0.45, 3, 5   # alarm: >= 3 of the last 5 frames at >= 0.45
SWEEP_PX, SWEEP_S = 0.30, 3.0                # wall inspection: the tip circles the lumen (fraction of W, period s)
INSPECT_AREA, INSPECT_S, COOLDOWN_S = 0.05, 1.5, 2.5   # a 6 mm polyp at the closest safe distance fills ~5-7%
END_S = 0.385

rng = np.random.default_rng(SEED)
stage = omni.usd.get_context().get_stage()
CONTROL = len(sys.argv) > 4 and sys.argv[4] == "control"   # no polyps: every alarm is a false alarm
cam, sensor, lesions = scene.build(stage, lesions=[] if CONTROL else None)
json.dump({"lesions": lesions, "fps": FPS, "W": W, "H": H, "hfov_deg": float(np.degrees(HFOV))}, open(f"{OUT}/scene.json", "w"))
omni.timeline.get_timeline_interface().play()

# centerline samples for wall-contact safety and progress
S = np.linspace(0, cg.L, 4000)
CP, CT, CN, CB = cg.frames(S)


class Detector:
    def __init__(self, port=5577):
        for _ in range(60):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=10); break
            except OSError:
                time.sleep(1)
        self.f = self.s.makefile("rb")

    def __call__(self, rgb):
        self.s.sendall(struct.pack("<II", rgb.shape[0], rgb.shape[1]) + np.ascontiguousarray(rgb).tobytes())
        return json.loads(self.f.readline())


def lumen_target(rgb):
    """Darkest blob of the blurred frame = the lumen far away (the headlight falls off with distance)."""
    small = Image.fromarray(rgb).convert("L").resize((80, 64), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    g = np.asarray(small, float)
    y, x = np.unravel_index(np.argmin(g), g.shape)
    return np.array([(x + 0.5) * W / 80, (y + 0.5) * H / 64])


def ray(px, fwd, up):
    """World direction through pixel px for a camera looking along fwd with image-up = up."""
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    ex = (px[0] - W / 2) / (W / 2) * np.tan(HFOV / 2)
    ey = (px[1] - H / 2) / (H / 2) * np.tan(VFOV / 2)
    d = fwd + ex * right - ey * up
    return d / np.linalg.norm(d)


def to_pixel(pt, pos, fwd, up):
    """World point -> image pixel for the tip camera (None if behind)."""
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    d = np.asarray(pt) - pos; z = d @ fwd
    if z <= 1e-4:
        return None
    return np.array([W / 2 + (d @ right) / z / np.tan(HFOV / 2) * W / 2, H / 2 - (d @ up) / z / np.tan(VFOV / 2) * H / 2])


def on_real_polyp(box, pos, fwd, up):
    """Ground truth for scoring only: does the alarm box sit on a real polyp (its projected top inside the box +50%)?"""
    if box is None:
        return False
    w, h = box[2] - box[0], box[3] - box[1]
    for l in lesions:
        px = to_pixel(l["top"], pos, fwd, up)
        if px is not None and box[0] - w / 2 <= px[0] <= box[2] + w / 2 and box[1] - h / 2 <= px[1] <= box[3] + h / 2:
            return True
    return False


def turn_towards(fwd, up, target_dir, max_angle):
    ang = np.arccos(np.clip(np.dot(fwd, target_dir), -1, 1))
    if ang < 1e-6:
        return fwd, up
    k = min(1.0, max_angle / ang)
    nf = fwd * (1 - k) + target_dir * k
    nf /= np.linalg.norm(nf)
    nu = up - np.dot(up, nf) * nf
    return nf, nu / np.linalg.norm(nu)


def nearest(pos):
    i = int(np.argmin(np.sum((CP - pos) ** 2, 1)))
    return i, S[i]


det = Detector()
i0 = int((0.015 + START_OFF) / cg.L * (len(S) - 1))
pos = CP[i0] + rng.normal(0, 0.002, 3)
fwd = CT[i0] / np.linalg.norm(CT[i0])
up = CB[i0] - np.dot(CB[i0], fwd) * fwd; up /= np.linalg.norm(up)
state, hold, cooldown, lost, stuck, blocked = "NAVIGATE", 0.0, 0.0, 0, 0, False
votes = []
finds, false_alarms, alarms, log = [], [], [], open(f"{OUT}/run.jsonl", "w")
t, step, wall_hits = 0.0, 0, 0
first = True
while True:
    rgb = scene.grab(app, cam, sensor, pos, scene.look_quat(fwd, up), settle=2, first=first)
    first = False
    r = det(rgb)
    best = max(r["boxes"], key=lambda b: b[4]) if r["boxes"] else None
    conf = best[4] if best else 0.0
    lum = lumen_target(rgb)
    alarm_ok = cooldown <= 0
    votes = (votes + [bool(best and conf >= ALARM_CONF and alarm_ok)])[-ALARM_WINDOW:]
    alarm = sum(votes) >= ALARM_VOTES

    if state == "NAVIGATE":
        ph = 2 * np.pi * t / SWEEP_S
        look = lum + SWEEP_PX * W * np.array([np.cos(ph), 0.8 * np.sin(ph)])   # circle the lumen to see the wall
        target = ray(np.clip(look, [0, 0], [W - 1, H - 1]), fwd, up)
        speed = V_NAV
        if alarm:
            state, lost = "APPROACH", 0
            alarms.append({"t": round(t, 2), "true": on_real_polyp(best[:4] if best else None, pos, fwd, up), "conf": round(conf, 3)})
    elif state == "APPROACH":
        if best and conf >= 0.25:
            lost = 0
            c = np.array([(best[0] + best[2]) / 2, (best[1] + best[3]) / 2])
            area = (best[2] - best[0]) * (best[3] - best[1]) / (W * H)
            target = ray(c, fwd, up)
            speed = V_APP if area < INSPECT_AREA else 0.0
            centred = np.linalg.norm(c - [W / 2, H / 2]) < 0.15 * W
            stuck = stuck + 1 if (blocked and centred) else 0       # pressed against the wall with the polyp centred
            if centred and (area >= INSPECT_AREA or stuck > 15):
                state, hold, stuck = "INSPECT", 0.0, 0
        else:
            lost += 1
            target, speed = ray(lum, fwd, up), V_APP
            if lost > 20:
                state, votes = "NAVIGATE", []
    else:  # INSPECT
        target, speed = fwd, 0.0
        hold += DT
        if hold >= INSPECT_S:
            # which real lesion is in view? (ground truth, for the score only; the robot never sees it)
            dists = [np.linalg.norm(np.array(l["top"]) - pos) for l in lesions]
            k = int(np.argmin(dists)) if dists else -1
            if k >= 0 and dists[k] < 0.04:                          # a real polyp within 4 cm is in front of the scope
                finds.append({"t": round(t, 2), "lesion": lesions[k]["id"], "class": lesions[k]["class"],
                              "size_mm": lesions[k]["size_mm"], "dist_mm": round(dists[k] * 1000, 1), "conf": round(conf, 3)})
            else:
                false_alarms.append({"t": round(t, 2), "s": round(s_now, 3), "conf": round(conf, 3)})
            print("INSPECTED", (finds or [None])[-1], len(false_alarms), "false alarms so far", flush=True)
            state, cooldown, votes = "NAVIGATE", COOLDOWN_S, []

    fwd, up = turn_towards(fwd, up, target, MAX_TURN * DT)
    pos = pos + fwd * speed * DT
    # wall contact: keep the tip inside the lumen (a real scope is pushed back by the wall)
    i, s_now = nearest(pos)
    off = pos - CP[i]; off -= np.dot(off, CT[i]) * CT[i]
    blocked = bool(np.linalg.norm(off) > 0.55 * cg.R0)
    if blocked:
        pos = CP[i] + off / np.linalg.norm(off) * 0.55 * cg.R0 + np.dot(pos - CP[i], CT[i]) * CT[i]
        wall_hits += 1

    Image.fromarray(rgb).save(f"{OUT}/frames/{step:05d}.jpg", quality=90)
    log.write(json.dumps({"i": step, "t": round(t, 3), "state": state, "pos": pos.round(5).tolist(), "fwd": fwd.round(4).tolist(),
                          "up": up.round(4).tolist(), "s": round(s_now, 4), "conf": round(conf, 3),
                          "box": [round(x, 1) for x in best[:4]] if best else None, "lumen": lum.tolist(), "ms": r["ms"]}) + "\n")
    t += DT; step += 1
    cooldown -= DT
    if s_now >= END_S or step > 2400:
        break

summary = {"run": RUN, "seed": SEED, "start_offset": START_OFF, "steps": step, "seconds": round(t, 2), "finds": finds,
           "found_ids": sorted({f["lesion"] for f in finds}), "lesions": [l["id"] for l in lesions], "wall_hits": wall_hits,
           "false_alarms": len(false_alarms), "false_alarm_list": false_alarms, "control": CONTROL,
           "alarms": len(alarms), "false_alarm_events": sum(not a["true"] for a in alarms), "alarm_list": alarms}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps(summary), flush=True)
log.close()
app.close()
