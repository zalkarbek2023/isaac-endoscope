# A robot endoscope that finds polyps by itself

A simulated robotic colonoscope drives through a **digital-twin colon in NVIDIA Isaac Sim**. Every RTX frame from its tip
camera goes to a **real polyp detector on TensorRT**; the scope follows the lumen, sweeps the wall, and steers to each alarm
until the polyp is centred for inspection.

**Result: 3 of 3 polyps found in 3 of 3 runs** (adenoma 6 mm, sessile serrated lesion 6 mm, hyperplastic 9 mm), ~48 s
for 40 cm of colon. The test polyps never appear in the detector's training data.

> Research demo, not a medical device. Polyp shapes and photos: REAL-Colon (CC BY 4.0).

## How it works

```
Isaac Sim 6.2 (Docker, headless, RTX)                         host
  digital-twin colon (colon_geom.py)                           detector_server.py
   40 cm tube, crescent haustral folds,                         YOLO26m polyp detector
   mucosa texture from a real colonoscopy frame,                TensorRT FP16, 960 px, ~7 ms/frame
   3 real polyps (Depth Anything V2 heightfield                 (localhost TCP)
   + SAM 2 mask + photo, REAL-Colon) on the
   outer wall of the 3 sharpest bends                    RGB ->
  tip camera ~110 deg + headlight (DiskLight child)      <- boxes
  endo_sim.py: finite-state autopilot
   NAVIGATE  follow the lumen (darkest region) + circle it to see the wall, 1.5 cm/s
   APPROACH  alarm = conf >= 0.45 on 3 of the last 5 frames -> steer to the box, 0.7 cm/s
   INSPECT   box >= 5% of the frame (or pressed to the wall) and centred -> hold 1.5 s
```

The scope is **kinematic**: insertion plus two-axis tip bending with a rate limit, and the wall pushes it back. There is no
soft-tissue physics.

## Results (`assets/summary_*.json`)

| Run | Start | Found | Alarms (false) | Time |
|---|---|---|---|---|
| sim_r1 | 1.5 cm | 3 / 3 (13.9, 24.2, 45.8 s) | 4 (1) | 49.3 s |
| sim_r2 | 2.5 cm | 3 / 3 (12.6, 22.1, 44.5 s) | 3 (0) | 48.4 s |
| sim_r3 | 3.5 cm | 3 / 3 (11.0, 22.0, 44.1 s) | 3 (0) | 48.0 s |
| control, no polyps | 1.5 cm | — | 2 (2), 1 false inspection | 31.7 s |

## What it took: the sim-to-real gap, honestly

1. The detector trained only on real colonoscopy (REAL-Colon) recognised rendered polyps at 1.5 cm (conf 0.56–0.74), but on
   the move, from 3–5 cm, it gave 0.3–0.5 and sometimes fired on folds: **0 of 3 found**.
2. Synthetic data from the twin (`synth_data.py`): 1,500 RTX frames of *other* polyps (shapes 0–2) at random places,
   labels from projecting the polyp mesh with occlusion checked against the rendered depth map.
   Two gotchas: the RTX camera annotator lags a pose jump by a few updates, and labels without the depth check land on
   polyps hidden behind folds.
3. Fine-tune (`finetune.py`): 1,500 synthetic + 1,500 real frames, 12 epochs from the real detector.
   **On the real validation videos mAP50 dropped from 0.472 to 0.384**, so this sim-adapted detector is used only inside
   the simulator.

## Run

```bash
# Linux + NVIDIA GPU (16 GB), Docker + NVIDIA Container Toolkit; REAL-Colon polyp3d assets and a YOLO polyp detector
./run_isaac.sh smoke.py                       # headless render check
./run_isaac.sh twin_render.py 5               # stills from inside the twin
./sim_train.sh 1500                           # synthetic frames -> fine-tune -> TensorRT engine
./eval_runs.sh out/polyp_sim.engine sim       # 3 runs + control
cd video && npm i && npx remotion render src/index.ts Robot out/robot.mp4 --gl=angle
```

Isaac Sim 6.2 notes: `RtxCamera(tick_rate=0)` never renders (use 30); fix the root-owned `~/docker/isaac-sim/cache/main/ov`
or the texture cache fails; a headlight of 3000 at metre scale looks like real colonoscopy.

## Credits

- NVIDIA Isaac Sim 6.2, TensorRT. Ultralytics YOLO26 (AGPL-3.0). Remotion, three.js.
- REAL-Colon: Biffi et al., *Scientific Data* 2024, CC BY 4.0 (polyp shapes, photos, mucosa texture, detector data).

Zalkarbek Arapbaev, physician and CV engineer.
