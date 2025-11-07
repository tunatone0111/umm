from .models.bagel_pipeline import BagelPipeline
import torch
import tqdm
import os
from .data import load_data


@torch.inference_mode()
def run():
    os.makedirs("out/winoground/images", exist_ok=True)

    pipe = BagelPipeline.from_pretrained(
        model_path="data/models/BAGEL-7B-MoT",  # Update this path
        device="cuda",
        max_memory_per_gpu="80GiB",
    )

    ds = load_data("winoground")

    for row in tqdm.tqdm(ds, total=len(ds)):
        pipe(
            images=[row["image_0"]],
            prompt=row["caption_0"],
            num_timesteps=50,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            max_image_size=512,
        ).save(f"out/winoground/images/{row['id']}_00.png")
        torch.cuda.empty_cache()

        pipe(
            images=[row["image_0"]],
            prompt=row["caption_1"],
            num_timesteps=50,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            max_image_size=512,
        ).save(f"out/winoground/images/{row['id']}_01.png")
        torch.cuda.empty_cache()

        pipe(
            images=[row["image_1"]],
            prompt=row["caption_0"],
            num_timesteps=50,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            max_image_size=512,
        ).save(f"out/winoground/images/{row['id']}_10.png")
        torch.cuda.empty_cache()

        pipe(
            images=[row["image_1"]],
            prompt=row["caption_1"],
            num_timesteps=50,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            max_image_size=512,
        ).save(f"out/winoground/images/{row['id']}_11.png")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
