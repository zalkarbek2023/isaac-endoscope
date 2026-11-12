"""Fine-tune the real polyp detector (polyp_v2, REAL-Colon) on synthetic Isaac Sim frames + real frames (no forgetting).

Train: out/synth (polyps 0-2 only; 3-5 stay held out for the autopilot) + 1500 random real REAL-Colon train frames.
Val:   300 real val frames (002-009, 003-002: the original validation videos) + 15% of synthetic frames.
Reports real-val mAP before and after; exports a TensorRT FP16 engine at 960.
"""
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

H = Path.home()
SYN = H / "robot-endo/out/synth"
REAL = H / "medai-colon/data/yolo_v2"
DS = H / "robot-endo/out/ds_sim"
BASE = H / "medai-colon/runs/detect/polyp_v2/weights/best.pt"
random.seed(0)
if DS.exists():
    shutil.rmtree(DS)
for sp in ("train", "val", "val_real"):
    (DS / sp / "images").mkdir(parents=True); (DS / sp / "labels").mkdir(parents=True)


def link(img, lab, sp):
    (DS / sp / "images" / img.name).symlink_to(img)
    dst = DS / sp / "labels" / (img.stem + ".txt")
    if lab.exists():
        dst.symlink_to(lab)
    else:
        dst.write_text("")


syn = sorted((SYN / "images").glob("*.jpg"))
random.shuffle(syn)
nv = int(0.15 * len(syn))
for i, im in enumerate(syn):
    link(im, SYN / "labels" / (im.stem + ".txt"), "val" if i < nv else "train")
real_tr = sorted((REAL / "images/train").glob("*.jpg")); random.shuffle(real_tr)
for im in real_tr[:1500]:
    link(im, REAL / "labels/train" / (im.stem + ".txt"), "train")
real_va = sorted((REAL / "images/val").glob("*.jpg")); random.shuffle(real_va)
for im in real_va[:300]:
    link(im, REAL / "labels/val" / (im.stem + ".txt"), "val")
    link(im, REAL / "labels/val" / (im.stem + ".txt"), "val_real")
(DS / "data.yaml").write_text(f"path: {DS}\ntrain: train/images\nval: val/images\nnames:\n  0: polyp\n")
(DS / "real.yaml").write_text(f"path: {DS}\ntrain: train/images\nval: val_real/images\nnames:\n  0: polyp\n")
print("train", len(list((DS / "train/images").iterdir())), "val", len(list((DS / "val/images").iterdir())), flush=True)

before = YOLO(str(BASE)).val(data=str(DS / "real.yaml"), imgsz=960, batch=8, verbose=False, plots=False)
print("REAL-VAL BEFORE mAP50", round(before.box.map50, 4), "mAP50-95", round(before.box.map, 4), flush=True)
m = YOLO(str(BASE))
m.train(data=str(DS / "data.yaml"), epochs=12, imgsz=960, batch=8, lr0=0.002, warmup_epochs=1, close_mosaic=3,
        project=str(H / "robot-endo/out/train"), name="polyp_sim", exist_ok=True, plots=False, verbose=True, workers=4)
best = H / "robot-endo/out/train/polyp_sim/weights/best.pt"
after = YOLO(str(best)).val(data=str(DS / "real.yaml"), imgsz=960, batch=8, verbose=False, plots=False)
print("REAL-VAL AFTER mAP50", round(after.box.map50, 4), "mAP50-95", round(after.box.map, 4), flush=True)
eng = YOLO(str(best)).export(format="engine", imgsz=960, half=True, device=0)
shutil.copy(eng, H / "robot-endo/out/polyp_sim.engine")
print("FINETUNE DONE", eng, flush=True)
