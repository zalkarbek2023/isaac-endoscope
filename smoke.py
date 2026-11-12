"""M0 smoke test: headless Isaac Sim 6.2 renders a cube -> out/smoke.png. Prints when the RTX camera starts delivering
non-black frames (the first run compiles shaders, so it can take hundreds of updates)."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 512})

import numpy as np
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.CreateSizeAttr(0.05)
cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.45, 0.35)])
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(800)

from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
cam = RtxCamera("/World/Cam", tick_rate=30.0, translations=np.array([0.0, 0.0, 0.15]), orientations=np.array([1.0, 0.0, 0.0, 0.0]))
cam.camera.set_clipping_ranges(0.001, 10.0)
light = UsdLux.DiskLight.Define(stage, "/World/Cam/Headlight")
light.CreateRadiusAttr(0.004); light.CreateIntensityAttr(3e5)
sensor = CameraSensor(cam, resolution=(512, 640), annotators=["rgb"])
omni.timeline.get_timeline_interface().play()
arr = None
for i in range(400):
    app.update()
    rgb, info = sensor.get_data("rgb")
    if i % 25 == 0:
        print("UPD", i, "none" if rgb is None else (rgb.shape, int(np.asarray(rgb.numpy()).max())), flush=True)
    if rgb is not None and np.asarray(rgb.numpy()).max() > 0:
        arr = np.asarray(rgb.numpy()); print("FIRST NONBLACK at update", i, flush=True); break
arr = arr if arr is not None else np.zeros((512, 640, 4), np.uint8)
Image.fromarray(arr[..., :3]).save("/work/out/smoke.png")
print("SMOKE mean", float(arr.mean()), "max", int(arr.max()), flush=True)
app.close()
