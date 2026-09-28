from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / "runs/t0_structure_audit_rectified_v2"
cameras = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE")
poses = ("zero", "mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm")
for pose in poses:
    panels = []
    for camera in cameras:
        path = root / pose / camera / "audit_panel.jpg"
        image = Image.open(path).convert("RGB")
        image.thumbnail((960, 150))
        panels.append((camera, image.copy()))
    width = 960
    height = sum(image.height + 24 for _, image in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for camera, image in panels:
        draw.text((4, y + 3), camera, fill=(0, 0, 0))
        canvas.paste(image, (0, y + 24))
        y += image.height + 24
    canvas.save(root / f"{pose}_seven_view_contact.jpg", quality=94)
print(root)
