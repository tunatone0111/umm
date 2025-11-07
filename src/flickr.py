import torch
import tqdm
import json
from PIL import Image
from argparse import ArgumentParser

from .models.bagel_pipeline import BagelPipeline

parser = ArgumentParser()
parser.add_argument("--dsmod", type=int, default=0)
args = parser.parse_args()


def run():
    with open("data/datasets/flickr8k/ExpertAnnotations.txt", "r") as f:
        experts = [line.strip().split("\t") for line in f.readlines()]
    with open("data/datasets/flickr8k/captions.json", "r") as f:
        captions = json.load(f)

    pipe = BagelPipeline.from_pretrained(
        model_path="data/models/BAGEL-7B-MoT",  # Update this path
        device="cuda",
        max_memory_per_gpu="80GiB",
    )

    for image_fname, caption_fname, e1, e2, e3 in tqdm.tqdm(
        experts, total=len(experts)
    ):
        caption_fname = caption_fname.split("#")[0]

        original = Image.open("data/datasets/flickr8k/images/" + image_fname)
        caption = captions[caption_fname][2]

        with torch.inference_mode():
            result = pipe(
                images=[original],
                prompt=caption,
                num_timesteps=50,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
            )

            output_image = result
            output_image.save(
                f"out/flickr8k/{image_fname.split('.')[0]}#{caption_fname.split('.')[0]}.png"
            )


if __name__ == "__main__":
    run()
