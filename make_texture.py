"""Wall texture from real colonoscopy: square mucosa crops (no polyp, inside the optics' octagon), mirror-tiled so the
texture repeats without seams. Writes candidates to out/tex/ and the chosen one to assets/mucosa.jpg."""
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

H = Path.home()
src = H / "medai-colon/outputs/polyp3d/web/hero_frame.jpg"
fr = H / "medai-colon/data/realcolon/frames/001-001/001-001_frames"
out = H / "robot-endo/out/tex"; out.mkdir(parents=True, exist_ok=True)
cands = [(src, (170, 180, 570, 580)), (src, (420, 120, 820, 520)),
         (fr / "001-001_23400.jpg", (500, 150, 900, 550)), (fr / "001-001_30000.jpg", (450, 250, 850, 650)),
         (fr / "001-001_15000.jpg", (450, 250, 850, 650)), (fr / "001-001_33000.jpg", (450, 250, 850, 650))]


def tile(im):
    a = im; b = ImageOps.mirror(im); c = ImageOps.flip(im); d = ImageOps.flip(b)
    t = Image.new("RGB", (im.width * 2, im.height * 2))
    t.paste(a, (0, 0)); t.paste(b, (im.width, 0)); t.paste(c, (0, im.height)); t.paste(d, (im.width, im.height))
    t = t.resize((1024, 1024), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=3, percent=80, threshold=2))
    return ImageEnhance.Contrast(t).enhance(1.35)   # bring out the vessels without shifting the hue


for k, (p, box) in enumerate(cands):
    if p.exists():
        tile(Image.open(p).convert("RGB").crop(box)).save(out / f"cand{k}.jpg", quality=90)
if len(sys.argv) > 1:
    (H / "robot-endo/assets").mkdir(exist_ok=True)
    Image.open(out / f"cand{sys.argv[1]}.jpg").save(H / "robot-endo/assets/mucosa.jpg", quality=92)
print(sorted(x.name for x in out.iterdir()))
