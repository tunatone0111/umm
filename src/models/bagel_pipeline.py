"""
Diffusers-style pipeline wrapper for BAGEL model with attention weights extraction.

This module provides a simple, clean interface for using the BAGEL model,
similar to how diffusers library pipelines work.

Example:
    from bagel_pipeline import BagelPipeline
    from PIL import Image

    # Load the pipeline
    pipe = BagelPipeline.from_pretrained("path/to/BAGEL-7B-MoT")

    # Edit an image
    image = Image.open("input.jpg")
    result = pipe(
        images=[image],
        prompt="make the sky blue",
        extract_attention_weights=True,
        save_attention_path="attention_weights.pkl"
    )

    # Access results
    edited_image = result["image"]
    attention_weights = result.get("attention_weights")
"""

import os
import torch
import copy
from typing import List, Optional, Union
from PIL import Image
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)

# Import BAGEL model components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bagel"))

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache


def move_generation_input_to_device(generation_input, device):
    """Utility to move all tensors in generation_input to device"""
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def apply_scale(width, height, scale):
    """Ensure dimensions are divisible by 16"""

    def _make_divisible(value, stride):
        return max(stride, int(round(value / stride) * stride))

    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height


class BagelPipeline:
    """
    Diffusers-style pipeline for BAGEL image editing model with attention extraction.

    This pipeline provides a simple interface for image editing with optional
    attention weights extraction from all layers and timesteps.
    """

    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        device="cuda",
    ):
        """
        Initialize the BAGEL pipeline.

        Args:
            model: The BAGEL model
            vae_model: VAE autoencoder model
            tokenizer: Qwen2 tokenizer
            vae_transform: Image transform for VAE
            vit_transform: Image transform for ViT
            new_token_ids: Special token IDs
            device: Device to run inference on
        """
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        max_memory_per_gpu: str = "80GiB",
        dtype: torch.dtype = torch.bfloat16,
        offload_folder: str = "offload",
    ):
        """
        Load a pretrained BAGEL model from a directory.

        Args:
            model_path: Path to the model directory
            device: Device to load the model on ("cuda" or "cpu")
            max_memory_per_gpu: Maximum memory per GPU (e.g., "80GiB")
            dtype: Data type for model weights (default: torch.bfloat16)
            offload_folder: Folder for offloading model weights

        Returns:
            BagelPipeline instance
        """
        print(f"Loading BAGEL model from {model_path}...")

        # Load configurations
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        # Load VAE
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_path, "ae.safetensors")
        )
        vae_model = vae_model.to(device).eval()

        # Create BAGEL config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
            extract_attention_weights=True,
        )

        # Initialize model with empty weights
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        # Load tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Create image transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Set up device map for multi-GPU
        device_map = infer_auto_device_map(
            model,
            max_memory={
                i: max_memory_per_gpu for i in range(torch.cuda.device_count())
            },
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        # Ensure certain modules are on the same device
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], device)
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = device
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Load checkpoint
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder=offload_folder,
            dtype=dtype,
            force_hooks=True,
        ).eval()

        print("Model loaded successfully!")

        return cls(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            device=device,
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        num_timesteps: int = 50,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = [0, 1.0],
        cfg_renorm_min: float = 0.0,
        cfg_type: str = "serial_text_img",
        cfg_renorm_type: str = "text_channel",
        timestep_shift: float = 3.0,
        max_image_size: int = 1024,
        min_image_size: int = 512,
        img_size: Optional[tuple] = None,
        extract_attention_weights: bool = False,
        extract_timesteps: Optional[List[int]] = None,
        save_attention_path: Optional[str] = None,
    ):
        """
        Edit an image using the BAGEL model.

        Args:
            images: Input image(s) (PIL Image or list of PIL Images)
            prompt: Text prompt describing the desired edit
            num_timesteps: Number of diffusion timesteps (default: 50)
            cfg_text_scale: Classifier-free guidance scale for text (default: 4.0)
            cfg_img_scale: Classifier-free guidance scale for image (default: 2.0)
            cfg_interval: CFG application interval (default: [0, 1.0])
            cfg_renorm_min: Minimum CFG renormalization (default: 0.0)
            cfg_type: Type of CFG ("serial_text_img" or "parallel")
            cfg_renorm_type: CFG renormalization type (default: "text_channel")
            timestep_shift: Timestep shift parameter (default: 3.0)
            max_image_size: Maximum image size (default: 1024)
            min_image_size: Minimum image size (default: 512)
            img_size: Explicit output size as (height, width)
            extract_attention_weights: Whether to extract attention weights (default: False)
            extract_timesteps: List of timestep indices to extract (e.g., [0, 10, 49]).
                              If None, extracts all timesteps (default: None)
            save_attention_path: Path to save attention weights pickle file

        Returns:
            If extract_attention_weights=False: PIL Image
            If extract_attention_weights=True: dict with keys "image" and "attention_weights"
        """
        # Ensure images is a list
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB
        images = [pil_img2rgb(img) for img in images]

        # Determine output size
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

        print(f"Processing image with size: H-{h} W-{w}")

        # Initialize KV cache
        past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens, new_rope = [0], [0]

        # Process input images
        for image in images:
            # Add VAE
            generation_input, newlens, new_rope = self.model.prepare_vae_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(
                generation_input, self.device
            )
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.model.forward_cache_update_vae(
                    self.vae_model, past_key_values, **generation_input
                )

            # Add ViT
            generation_input, newlens, new_rope = self.model.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(
                generation_input, self.device
            )
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input
                )

        # Prepare CFG text branch
        cfg_text_past_key_values = copy.deepcopy(past_key_values)
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
        )
        generation_input_cfg_text = move_generation_input_to_device(
            generation_input_cfg_text, self.device
        )

        # Prepare CFG img branch
        cfg_img_past_key_values = NaiveCache(
            self.model.config.llm_config.num_hidden_layers
        )
        cfg_img_newlens = [0]
        cfg_img_new_rope = [0]
        generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = (
            self.model.prepare_prompts(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                prompts=[prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
        )
        generation_input_cfg_img = move_generation_input_to_device(
            generation_input_cfg_img, self.device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_img_past_key_values = self.model.forward_cache_update_text(
                cfg_img_past_key_values, **generation_input_cfg_img
            )
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            image_sizes=[(h, w)],
        )
        generation_input_cfg_img = move_generation_input_to_device(
            generation_input_cfg_img, self.device
        )

        # Prepare main branch with prompt
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_generation_input_to_device(
            generation_input, self.device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(
                past_key_values, **generation_input
            )
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_generation_input_to_device(
            generation_input, self.device
        )

        # Generate image
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.model.generate_image(
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
                extract_attention_weights=extract_attention_weights,
                extract_timesteps=extract_timesteps,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text[
                    "cfg_packed_position_ids"
                ],
                cfg_text_packed_query_indexes=generation_input_cfg_text[
                    "cfg_packed_query_indexes"
                ],
                cfg_text_key_values_lens=generation_input_cfg_text[
                    "cfg_key_values_lens"
                ],
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

        # Decode latent
        latent = unpacked_latent[0]
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        output_image = self.vae_model.decode(latent)
        output_image = (
            ((output_image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        output_image = Image.fromarray(output_image)

        # Handle attention weights
        if extract_attention_weights:
            import pickle
            from datetime import datetime

            attention_weights = self.model.attention_weights_storage

            if save_attention_path:
                # Save to the specified path
                with open(save_attention_path, "wb") as f:
                    pickle.dump(attention_weights, f)
                print(f"Attention weights saved to {save_attention_path}")
            else:
                # Auto-generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                auto_path = f"attention_weights_{timestamp}.pkl"
                with open(auto_path, "wb") as f:
                    pickle.dump(attention_weights, f)
                print(f"Attention weights automatically saved to {auto_path}")

            return {"image": output_image, "attention_weights": attention_weights}
        else:
            return output_image
