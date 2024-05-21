from typing import Any, Dict,Tuple
from abc import ABC, abstractmethod
from collections.abc import Iterable 
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
from torch import Tensor

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL

import numpy as np
import yaml
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transform


default_prompt = "dashcam recording, urban driving scene, video, autonomous driving, detailed cars, traffic scene, pandaset, kitti, high resolution, realistic, detailed, camera video, dslr, ultra quality, sharp focus, crystal clear, 8K UHD, 10 Hz capture frequency 1/2.7 CMOS sensor, 1920x1080"
default_negative_prompt = "face, human features, unrealistic, artifacts, blurry, noisy image, NeRF, oil-painting, art, drawing, poor geometry, oversaturated, undersaturated, distorted, bad image, bad photo"


def get_device():
    # TODO: Make compatible with multi-gpu if possible

    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def batch_if_not_iterable(item: any, single_dim: int = 3) -> Iterable[Any]:
    if item is None:
        return item

    if isinstance(item, (torch.Tensor, np.ndarray)):
        if len(item) == single_dim:
            item = item[None, ...]

        return item

    if not isinstance(item, Iterable):
        return [item]

    return item


def validate_same_len(*iters) -> None:
    prev_iter_len = None
    for iterator in iters:
        if iterator is None:
            continue

        iter_len = len(iterator)

        if (prev_iter_len is not None) and (iter_len != prev_iter_len):
            raise ValueError(
                f"Expected same length on iterators, but received {[len(i) for i in iters]}"
            )

        prev_iter_len = iter_len


def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


norm_img_pipeline = transform.Compose([transform.ConvertImageDtype(torch.float32)])


def read_image(
    img_path: Path, tf_pipeline: transform.Compose = norm_img_pipeline
) -> Tensor:
    img = torchvision.io.read_image(str(img_path))
    img = tf_pipeline(img)

    return img



def load_diffusion_model(
    model_config_params: Dict[str, Any], device=get_device()
) -> "DiffusionModel":
    logging.info(f"Loading diffusion model...")

    model_name = model_config_params.get("model_name")
    constructor = model_name_to_constructor.get(model_name)
    if not constructor:
        raise NotImplementedError

    model = constructor(configs=model_config_params, device=device)
    logging.info(f"Finished loading diffusion model")
    return model


@dataclass
class ModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"
    mock = "mock"


sdxl_models = {
    ModelId.sdxl_base_v1_0,
    ModelId.sdxl_refiner_v1_0,
    ModelId.sdxl_turbo_v1_0,
}
sd_models = {ModelId.sd_v1_5, ModelId.mock}


def is_sdxl_model(model_id: str) -> bool:
    return model_id in sdxl_models


def is_sdxl_vae(model_id: str) -> bool:
    return model_id == "madebyollin/sdxl-vae-fp16-fix" or is_sdxl_model(model_id)


def prep_model(
    pipe, device=get_device(), low_mem_mode: bool = False, compile: bool = True
):
    if compile:
        try:
            pipe.unet = torch.compile(pipe.unet, fullgraph=True)
        except AttributeError:
            logging.warn(f"No unet found in Pipe. Skipping compiling")

    if low_mem_mode:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


class DiffusionModel(ABC):
    @abstractmethod
    def diffuse_sample(self, sample: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class MockPipe(DiffusionModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def diffuse_sample(
        self,
        sample: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        image = sample["rgb"]

        if len(image.shape) == 3:
            image = image[None, ...]

        return {
            "rgb": image
        }

class SDPipe(DiffusionModel):
    def __init__(
        self,
        configs: Dict[str, Any] = None,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()
        if configs is None:
            configs = {}

        model_id = configs.get("model_id", ModelId.sd_v2_1)
        low_mem_mode = configs.get("low_mem_mode", False)
        compile_model = configs.get("compile_model", False)

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        if lora_weights := configs.get("lora_weights"):
            self.pipe.load_lora_weights(lora_weights)

        self.pipe = prep_model(
            self.pipe,
            low_mem_mode=low_mem_mode,
            device=device,
            compile=compile_model,
        )
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        self.device = device

    def diffuse_sample(
        self,
        sample: Dict[str, Any],
        strength: float = 0.2,
        denoising_start: int = None,
        denoising_end: int = None,
        original_size: Tuple[int, int] = (1024, 1024),
        target_size: Tuple[int, int] = (1024, 1024),
        prompt: str = default_prompt,
        negative_prompt: str = default_negative_prompt,
        num_steps: int = 50,
        gen: torch.Generator | Iterable[torch.Generator] = None,
        kwargs: Dict[str, Any] = None,
    ):
        image = sample["rgb"]

        if len(image.shape) == 3:
            image = image[None, ...]

        batch_size = len(image)

        if image.size(1) == 3:
            channel_first = True
        elif image.size(3) == 3:
            channel_first = False
        else:
            raise ValueError(f"Image needs to be BCHW or BHWC, received {image.shape}")

        if not channel_first:        
            image = image.permute(0, 3, 1, 2)       # Diffusion model is channel first

        kwargs = kwargs or {}
        if prompt is not None:
            with torch.no_grad():
                tokens = tokenize_prompt(self.pipe.tokenizer, prompt).to(self.device)
                prompt_embeds = encode_tokens(
                    self.pipe.text_encoder, tokens, using_sdxl=False
                )["embeds"]
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)

        if negative_prompt is not None:
            with torch.no_grad():
                negative_tokens = tokenize_prompt(self.pipe.tokenizer, negative_prompt).to(
                    self.device
                )
                negative_prompt_embeds = encode_tokens(
                    self.pipe.text_encoder, negative_tokens, using_sdxl=False
                )["embeds"]

            negative_prompt_embeds = negative_prompt_embeds.expand(batch_size, -1, -1)

        image = self.pipe(
            image=image,
            generator=gen,
            output_type="pt",
            strength=strength,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            num_inference_steps=num_steps,
            original_size=original_size,
            target_size=target_size,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            **kwargs,
        ).images

        if not channel_first:
            image = image.permute(0, 2, 3, 1)

        return {"rgb": image}



model_name_to_constructor = {
    "sd_base": SDPipe,
    "mock": MockPipe,
}


def encode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, img: Tensor, seed: int = 0
) -> Tensor:
    img = img_processor.preprocess(img)

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to("cuda"))

    if needs_upcasting:
        vae.to(original_vae_dtype)

    latents = retrieve_latents(latents, generator=torch.manual_seed(seed))
    latents = latents * vae.config.scaling_factor

    return latents


def decode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, latents: Tensor
) -> Tensor:
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    img = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    if needs_upcasting:
        vae.to(original_vae_dtype)

    img = img_processor.postprocess(img, output_type="pt")
    return img


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    vae.post_quant_conv.to(dtype)
    vae.decoder.conv_in.to(dtype)
    vae.decoder.mid_block.to(dtype)

    return vae




def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = text_inputs.input_ids
    return tokens


def encode_tokens(text_encoder, tokens, using_sdxl):
    if using_sdxl:
        raise NotImplementedError

    prompt_embeds = text_encoder(tokens)

    return {"embeds": prompt_embeds.last_hidden_state}
