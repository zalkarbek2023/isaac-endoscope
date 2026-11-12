"""Perception node: our real polyp detector (colonoscopy-ai YOLO26m, 960 px, TensorRT FP16) served over localhost TCP.

The Isaac Sim script (inside Docker, --network=host) sends raw RGB frames; this host process answers with boxes.
Protocol: request = 8-byte header (uint32 H, uint32 W, little-endian) + H*W*3 bytes RGB; reply = one JSON line
{"boxes": [[x1, y1, x2, y2, conf], ...], "ms": inference_ms}.
Usage: python detector_server.py            # serve on 127.0.0.1:5577
       python detector_server.py --test a.png b.png   # print detections for stills (domain-gap check)
"""
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ENGINE=~/robot-endo/out/polyp_sim.engine for the detector fine-tuned on synthetic Isaac frames
ENGINE = Path(os.environ.get("ENGINE", Path.home() / "medai-colon/models/polyp_v2.engine")).expanduser()
model = YOLO(str(ENGINE), task="detect")


def detect(rgb):
    t0 = time.perf_counter()
    r = model.predict(rgb[..., ::-1].copy(), imgsz=960, conf=0.10, half=True, verbose=False)[0]   # ultralytics wants BGR
    ms = (time.perf_counter() - t0) * 1000
    boxes = [[*map(float, b.xyxy[0].tolist()), float(b.conf[0])] for b in r.boxes]
    return boxes, ms


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def serve(port=5577):
    detect(np.zeros((512, 640, 3), np.uint8))                     # warm up the engine
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(1)
    print(f"detector ready on 127.0.0.1:{port} ({ENGINE.name})", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                h, w = struct.unpack("<II", recv_exact(conn, 8))
                rgb = np.frombuffer(recv_exact(conn, h * w * 3), np.uint8).reshape(h, w, 3)
                boxes, ms = detect(rgb)
                conn.sendall((json.dumps({"boxes": boxes, "ms": round(ms, 2)}) + "\n").encode())
        except ConnectionError:
            conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        from PIL import Image
        for p in sys.argv[2:]:
            boxes, ms = detect(np.asarray(Image.open(p).convert("RGB")))
            best = max((b[4] for b in boxes), default=0.0)
            print(f"{Path(p).name}: best conf {best:.3f}, {len(boxes)} boxes, {ms:.1f} ms", flush=True)
    else:
        serve()
