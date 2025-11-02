import torch
from diffusers import QwenImageEditPlusPipeline


def load_qwen_image_edit():
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)

    return pipe
