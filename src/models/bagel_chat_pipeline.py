"""
Diffusers-style pipeline wrapper for BAGEL model in chat/VQA mode.

This module provides a simple, clean interface for using the BAGEL model
for visual question answering and chat tasks, similar to how diffusers
library pipelines work.

Example:
    from bagel_chat_pipeline import BagelChatPipeline
    from PIL import Image

    # Load the pipeline
    pipe = BagelChatPipeline.from_pretrained("path/to/BAGEL-7B-MoT")

    # Chat with an image
    image = Image.open("input.jpg")
    response = pipe(
        images=[image],
        prompt="What is in this image?",
        max_length=100
    )
    print(response)

    # Use chain of thought mode
    response = pipe(
        images=[image],
        prompt="Is there a cat in this image?",
        cot=True,
        max_length=1000
    )
    print(response)  # Returns extracted answer
"""

import os
import re
import torch
from typing import List, Optional, Union, Dict
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


# Chain of Thought instruction template
COT_INSTRUCTION = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    'please use the format "Final answer: .."'
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


def extract_answer(text: str) -> str:
    """
    Extract the final answer from chain of thought response.

    Args:
        text: The full response text with reasoning

    Returns:
        Extracted answer or original text if no answer pattern found
    """
    match = re.search(r"(Final answer:|Answer:)\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text


class BagelChatPipeline:
    """
    Diffusers-style pipeline for BAGEL chat/VQA model.

    This pipeline provides a simple interface for visual question answering
    and chat tasks with the BAGEL model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        vit_transform,
        new_token_ids,
        device="cuda",
    ):
        """
        Initialize the BAGEL chat pipeline.

        Args:
            model: The BAGEL model (in visual understanding mode)
            tokenizer: Qwen2 tokenizer
            vit_transform: Image transform for ViT
            new_token_ids: Special token IDs
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
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
        transform_config_path: Optional[str] = None,
    ):
        """
        Load a pretrained BAGEL model from a directory for chat/VQA tasks.

        Args:
            model_path: Path to the model directory
            device: Device to load the model on ("cuda" or "cpu")
            max_memory_per_gpu: Maximum memory per GPU (e.g., "80GiB")
            dtype: Data type for model weights (default: torch.bfloat16)
            offload_folder: Folder for offloading model weights
            transform_config_path: Path to config file for image transform parameters.
                                  If None, uses default values.

        Returns:
            BagelChatPipeline instance
        """
        print(f"Loading BAGEL model in chat mode from {model_path}...")

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

        # Create BAGEL config for chat/VQA mode (visual understanding only)
        config = BagelConfig(
            visual_gen=False,  # No image generation
            visual_und=True,  # Visual understanding enabled
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
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

        # Create image transform for ViT
        if transform_config_path and os.path.exists(transform_config_path):
            import yaml

            with open(transform_config_path, "r") as f:
                data_config = yaml.safe_load(f)

            transform_args = data_config["vlm_sft"]["image_transform_args"]
            vit_transform = ImageTransform(
                max_image_size=transform_args["max_image_size"],
                min_image_size=transform_args["min_image_size"],
                image_stride=transform_args["image_stride"],
                max_pixels=transform_args.get("max_pixels"),
            )
        else:
            # Use default values
            vit_transform = ImageTransform(
                max_image_size=980,
                min_image_size=224,
                image_stride=14,
            )

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

        print("Model loaded successfully in chat mode!")

        return cls(
            model=model,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            device=device,
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        max_length: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        cot: bool = False,
        return_full_reasoning: bool = False,
    ) -> Union[str, Dict[str, str]]:
        """
        Perform chat/VQA inference with the BAGEL model.

        Args:
            images: Input image(s) (PIL Image or list of PIL Images)
            prompt: Text prompt/question
            max_length: Maximum number of tokens to generate (default: 100)
            do_sample: Whether to use sampling for generation (default: False)
            temperature: Sampling temperature (default: 1.0)
            cot: Enable chain of thought mode for step-by-step reasoning (default: False)
                 When enabled, max_length is automatically increased to 1000 if not explicitly set higher
            return_full_reasoning: If True and cot=True, returns dict with both 'answer' and
                                  'reasoning'. If False and cot=True, returns only extracted answer.
                                  Ignored when cot=False (default: False)

        Returns:
            If cot=False or (cot=True and return_full_reasoning=False):
                Generated text response as a string
            If cot=True and return_full_reasoning=True:
                Dictionary with 'answer' (extracted) and 'reasoning' (full response)
        """
        # Ensure images is a list
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB
        images = [pil_img2rgb(img) for img in images]

        # Apply chain of thought instruction if enabled
        if cot:
            prompt = COT_INSTRUCTION.format(question=prompt)
            # Automatically increase max_length for CoT if not explicitly set
            if max_length == 100:  # Default value
                max_length = 1000

        # Call the model's chat method
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            response = self.model.chat(
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
                image_transform=self.vit_transform,
                images=images,
                prompt=prompt,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )

        # Extract answer from CoT response if enabled
        if cot:
            extracted = extract_answer(response)
            if return_full_reasoning:
                return {"answer": extracted, "reasoning": response}
            return extracted

        return response
