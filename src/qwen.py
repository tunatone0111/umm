import torch
import os
import requests
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser

from .data import load_data
from diffusers import QwenImageEditPlusPipeline

parser = ArgumentParser()
parser.add_argument("--dsmod", type=int, default=0)
args = parser.parse_args()


def run():
    dsmod = args.dsmod
    dname = "datacomp_small"
    ds = load_data(dname)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    os.makedirs(f"out/{dname}/qwen", exist_ok=True)

    for i, row in enumerate(ds):
        if os.path.exists(f"out/{dname}/qwen/{row['uid']}_origin.png"):
            continue
        if i % 4 != dsmod:
            continue

        prompt = row["text"]
        try:
            response = requests.get(row["url"])
            original = Image.open(BytesIO(response.content))
            original.convert("RGB")
        except Exception as e:
            print(f"Error loading image {row['uid']}: {e}")
            continue
        except OSError as e:
            print(f"Error converting image {row['uid']}: {e}")
            continue

        with torch.inference_mode():
            result = pipe(
                image=[original],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=40,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cuda").manual_seed(0),
            )

        output_image = result.images[0]
        with open(f"out/{dname}/qwen/{row['uid']}_prompt.txt", "w") as f:
            f.write(prompt)
        original.save(f"out/{dname}/qwen/{row['uid']}_origin.png")
        output_image.save(f"out/{dname}/qwen/{row['uid']}_edited.png")


if __name__ == "__main__":
    run()
