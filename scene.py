"""Shared Isaac Sim scene for the robot endoscope (import only AFTER SimulationApp has been created).

build(stage, pidx, ...) puts the colon twin (wall + one real polyp) and the endoscope tip camera with its headlight on
the stage and returns (camera, sensor, polyp meta, polyp top point, polyp inward normal).
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

import colon_geom as cg

W, H = 640, 512


def material(stage, path, tex, rough=0.22, spec=0.6):
    """UsdPreviewSurface with a diffuse texture; low roughness = wet mucosa highlights."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(spec, spec, spec))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, path + "/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("mirror")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("mirror")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def mesh(stage, path, v, uv, f, mat, nrm=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1).astype(np.int32)))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy((cg.normals(v, f) if nrm is None else nrm).astype(np.float32)))
    m.SetNormalsInterpolation("vertex")
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def look_quat(fwd, up):
    """wxyz quaternion turning a USD camera (looks down -Z, +Y up) to look along fwd with the given up hint."""
    f = np.asarray(fwd, float); f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], 1)                                    # columns = camera X, Y, Z in world coords
    q = Gf.Matrix3d(*m.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return [q.GetReal(), *q.GetImaginary()]


# default lesions for the autopilot: (polyp3d index, arc position s [m], angle around the wall [rad])
LESIONS = cg.LESIONS   # AD 6 mm, SSL 6 mm, HP 9 mm on the outer wall of the 3 sharpest bends


def build(stage, pidx=None, light_intensity=3000.0, s_p=0.30, th_p=1.9, focal=7.3, lesions=None, depth=False):
    """pidx: one polyp at (s_p, th_p) (stills / tests); else `lesions` (default LESIONS) for the autopilot."""
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    v, uv, f, nrm = cg.tube()
    mesh(stage, "/World/Colon/Wall", v, uv, f, material(stage, "/World/Looks/Mucosa", "/work/assets/mucosa.jpg"), nrm)
    lesions = [(pidx, s_p, th_p)] if pidx is not None else (LESIONS if lesions is None else lesions)   # [] = no polyps (control run)
    found = []
    for k, (idx, sp, tp) in enumerate(lesions):
        pv, puv, pf, top, inward, meta = cg.polyp("/assets/polyp3d", idx, s_p=sp, th_p=tp)
        mesh(stage, f"/World/Colon/Polyp{k}", pv, puv, pf,
             material(stage, f"/World/Looks/Polyp{k}", f"/assets/polyp3d/p{idx}.jpg", 0.18, 0.7))
        found.append({"idx": idx, "id": meta["id"], "class": meta["class"], "size_mm": meta["size_mm"],
                      "s": sp, "top": top.tolist(), "inward": inward.tolist()})
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
    cam = RtxCamera("/World/Endo", tick_rate=30.0)   # 0 never renders in 6.2
    ucam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Endo"))
    ucam.GetHorizontalApertureAttr().Set(20.955)
    ucam.GetFocalLengthAttr().Set(focal)                           # 7.3 mm ~ 110 deg horizontal FOV
    ucam.GetClippingRangeAttr().Set(Gf.Vec2f(0.0005, 2.0))
    light = UsdLux.DiskLight.Define(stage, "/World/Endo/Headlight")
    light.CreateRadiusAttr(0.003)
    light.CreateIntensityAttr(light_intensity)
    sensor = CameraSensor(cam, resolution=(H, W), annotators=["rgb", "distance_to_image_plane"] if depth else ["rgb"])
    if pidx is not None:
        return cam, sensor, meta, top, inward
    return cam, sensor, found


def grab(app, cam, sensor, pos, quat, settle=3, first=False):
    """Move the tip camera and return an RGB uint8 frame (a few updates let temporal AA settle after a jump)."""
    cam.set_world_poses(positions=[list(map(float, pos))], orientations=[list(map(float, quat))])
    rgb = None
    for i in range(40 if first else settle + 6):
        app.update()
        if i >= (12 if first else settle):
            rgb, _ = sensor.get_data("rgb")
            if rgb is not None and np.asarray(rgb.numpy()).max() > 0:
                break
    return np.asarray(rgb.numpy())[..., :3].copy() if rgb is not None else np.zeros((H, W, 3), np.uint8)


def depth_map(sensor):
    """Metres from the image plane per pixel (needs build(..., depth=True)); call right after grab()."""
    d, _ = sensor.get_data("distance_to_image_plane")
    return None if d is None else np.asarray(d.numpy()).reshape(H, W)
