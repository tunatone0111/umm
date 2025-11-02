from diffusers import DiffusionPipeline
from PIL import Image
import torch


def load_bagel():
    pipe = DiffusionPipeline.from_pretrained(
        "JiaxinGe/Diffusers-BAGEL",
        custom_pipeline="JiaxinGe/Diffusers-BAGEL",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)

    return pipe


def apply_scale(width, height, scale):
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height


def editing_image(
    images,
    prompt,
    num_timesteps=50,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0, 1.0],
    cfg_renorm_min=0.0,
    cfg_type="serial_text_img",
    cfg_renorm_type="text_channel",
    timestep_shift=3.0,
    max_image_size=1024,
    min_image_size=512,
    img_size=None,
    device=None,
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=vae_transform,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(
                vae_model, past_key_values, **generation_input
            )

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

    # cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(h, w)],
    )
    generation_input_cfg_text = move_generation_input_to_device(
        generation_input_cfg_text, device
    )
    # cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = (
        gen_model.prepare_prompts(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
    )
    generation_input_cfg_img = move_generation_input_to_device(
        generation_input_cfg_img, device
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(
            cfg_img_past_key_values, **generation_input_cfg_img
        )
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        image_sizes=[(h, w)],
    )
    generation_input_cfg_img = move_generation_input_to_device(
        generation_input_cfg_img, device
    )

    # origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(
            past_key_values, **generation_input
        )
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(h, w)],
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h // 8, w // 8)
    tmpimage = vae_model.decode(latent)
    tmpimage = (
        ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    tmpimage = Image.fromarray(tmpimage)

    return tmpimage


def run(images: list[Image.Image], prompt: str):
    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [0.0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    tmpimage = editing_image(
        images=images,
        prompt=prompt,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=cfg_interval,
        cfg_renorm_min=cfg_renorm_min,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        device="cuda",
    )

    tmpimage = tmpimage.crop(tmpimage.getbbox())
    tmpimage.save(outpath)
