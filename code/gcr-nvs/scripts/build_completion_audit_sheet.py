"""Build a seven-view source/warp/completion/target audit sheet."""

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/media/2T_HD/GCR-NVS_DATA/outputs/generative_route/eval/completion_final_7views_960")
CAMERAS = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE")

rows = []
for camera in CAMERAS:
    pred = next(ROOT.glob(f"*_{camera}_prediction.png"), None)
    if pred is None:
        continue
    stem = pred.name[:-len("_prediction.png")]
    paths = [ROOT / f"{stem}_{suffix}.png" for suffix in ("warp", "prediction", "target", "validity")]
    images = [Image.open(path).convert("RGB") for path in paths]
    w = 360
    h = round(images[0].height * w / images[0].width)
    panel = Image.new("RGB", (w * 3, h * 2 + 42), "white")
    draw = ImageDraw.Draw(panel)
    for i, (image, label) in enumerate(zip(images[:3], ("T0 warp", "completion", "target"))):
        panel.paste(image.resize((w, h), Image.Resampling.LANCZOS), (i * w, 34))
        draw.text((i * w + 8, 10), label, fill="black")
    mask = images[3].resize((w, h), Image.Resampling.NEAREST)
    panel.paste(mask, (0, h + 42))
    draw.text((8, h + 50), "validity", fill="black")
    rows.append((camera, panel))

if rows:
    canvas = Image.new("RGB", (rows[0][1].width, sum(p.height for _, p in rows)), "#e9e9e9")
    y = 0
    draw = ImageDraw.Draw(canvas)
    for camera, panel in rows:
        canvas.paste(panel, (0, y))
        draw.text((8, y + 3), camera, fill="#e11d48")
        y += panel.height
    output = ROOT / "source_t0_completion_target_audit.jpg"
    canvas.save(output, quality=94)
    print(output)
