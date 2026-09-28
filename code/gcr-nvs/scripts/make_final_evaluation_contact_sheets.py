from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/media/2T_HD/GCR-NVS_DATA/outputs/generative_route/evaluations')

def make(name):
    src = ROOT / name
    rows = sorted(src.glob('*_prediction.png'))
    panels = []
    for pred in rows:
        stem = pred.name[:-len('_prediction.png')]
        target = src / f'{stem}_target.png'
        warp = src / f'{stem}_warp.png'
        images = [Image.open(p).convert('RGB') for p in (pred, target, warp)]
        w = min(480, images[0].width)
        h = round(images[0].height * w / images[0].width)
        images = [im.resize((w, h), Image.Resampling.LANCZOS) for im in images]
        panel = Image.new('RGB', (w * 3, h + 34), 'white')
        labels = ['prediction', 'target', 'warp']
        for i, (im, label) in enumerate(zip(images, labels)):
            panel.paste(im, (i * w, 34))
            ImageDraw.Draw(panel).text((i * w + 8, 8), label, fill='black')
        panels.append((stem, panel))
    if not panels:
        return
    width = panels[0][1].width
    height = sum(p.height for _, p in panels)
    canvas = Image.new('RGB', (width, height), '#eeeeee')
    y = 0
    for stem, panel in panels:
        canvas.paste(panel, (0, y)); y += panel.height
    out = src / 'final_contact_sheet.jpg'
    canvas.save(out, quality=92)
    print(out)

for name in ('appearance_vitl_full_strict_loo', 'appearance_vitl_full_pose_reprojection'):
    make(name)
