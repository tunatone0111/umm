from diffusers import QwenImageEditPlusPipeline
import torch
import tqdm
import os
from .data import load_data


@torch.inference_mode()
def run():
    os.makedirs("out/winoground_qwen/images", exist_ok=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    ds = load_data("winoground")

    for row in tqdm.tqdm(ds, total=len(ds)):
        pipe(
            image=[row["image_0"]],
            prompt=row["caption_0"],
            negative_prompt=" ",
            num_inference_steps=40,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
            height=512,
            width=512,
        ).images[0].save(f"out/winoground_qwen/images/{row['id']}_00.png")

        pipe(
            image=[row["image_0"]],
            prompt=row["caption_1"],
            negative_prompt=" ",
            num_inference_steps=40,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
            height=512,
            width=512,
        ).images[0].save(f"out/winoground_qwen/images/{row['id']}_01.png")

        pipe(
            image=[row["image_1"]],
            prompt=row["caption_0"],
            negative_prompt=" ",
            num_inference_steps=40,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
            height=512,
            width=512,
        ).images[0].save(f"out/winoground_qwen/images/{row['id']}_10.png")

        pipe(
            image=[row["image_1"]],
            prompt=row["caption_1"],
            negative_prompt=" ",
            num_inference_steps=40,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
            height=512,
            width=512,
        ).images[0].save(f"out/winoground_qwen/images/{row['id']}_11.png")


if __name__ == "__main__":
    run()
