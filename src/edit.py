import torch
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
import random

from .data import load_data
from .models.bagel_pipeline import BagelPipeline

dname = "hq_edit"
ds = load_data(dname)
pipe = BagelPipeline.from_pretrained(
    model_path="data/models/BAGEL-7B-MoT",  # Update this path
    device="cuda",
    max_memory_per_gpu="80GiB",
)


random_index = random.randint(0, len(ds) - 1)
row = ds[random_index]

if dname == "hq_edit":
    prompt = row["edit"]
    original = row["input_image"]
else:
    prompt = row["text"]
    response = requests.get(row["url"])
    original = Image.open(BytesIO(response.content))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with torch.inference_mode():
    result = pipe(
        images=[original],
        prompt=prompt,
        num_timesteps=50,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        extract_attention_weights=True,
        extract_timesteps=[5, 30, 48],
        save_attention_path=f"{dname}_attention_weights_{timestamp}.pkl",
    )

    output_image = result["image"]
    with open(f"{dname}_prompt_{timestamp}.txt", "w") as f:
        f.write(prompt)
    original.save(f"{dname}_original_{timestamp}.png")
    output_image.save(f"{dname}_edited_{timestamp}.png")
