from .metrics.sscd import measure_SSCD_similarity
from .utils import apply_scale
from .data import load_data
from PIL import Image

ds = load_data("winoground")

text_acc = 0
image_acc = 0
group_acc = 0

for row in ds:
    x0 = row["image_0"]
    x1 = row["image_1"]
    x00 = Image.open(f"out/winoground/images/{row['id']}_00.png")
    x01 = Image.open(f"out/winoground/images/{row['id']}_01.png")
    x10 = Image.open(f"out/winoground/images/{row['id']}_10.png")
    x11 = Image.open(f"out/winoground/images/{row['id']}_11.png")

    w, h = x0.size
    scale = 512 / max(w, h)
    w, h = apply_scale(w, h, scale)

    x0 = x0.resize((w, h))
    x1 = x1.resize((w, h))

    sscd = measure_SSCD_similarity([x0, x1], [x00, x01, x10, x11])
    s00 = sscd[0, 0]
    s01 = sscd[0, 1]
    s10 = sscd[1, 0]
    s11 = sscd[1, 1]

    text = (s00 > s01) and (s11 > s10)
    image = (s00 > s10) and (s11 > s01)
    group = text and image

    text_acc += text
    image_acc += image
    group_acc += group

    print(
        f"[{row['id'] + 1}/{len(ds)}] {text_acc / (row['id'] + 1):.2%}, {image_acc / (row['id'] + 1):.2%}, {group_acc / (row['id'] + 1):.2%}"
    )


print(
    f"Accuracy: {text_acc / len(ds):.2%}, {image_acc / len(ds):.2%}, {group_acc / len(ds):.2%}"
)
