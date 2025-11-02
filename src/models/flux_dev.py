import torch
from diffusers import FluxPipeline


def load_flux_dev():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)

    return pipe
