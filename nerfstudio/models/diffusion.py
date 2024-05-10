from typing import Any
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


def batch_if_not_iterable(item: any, single_dim: int = 3) -> Iterable[any]:
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


@dataclass
class ModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"


sdxl_models = {
    ModelId.sdxl_base_v1_0,
    ModelId.sdxl_refiner_v1_0,
    ModelId.sdxl_turbo_v1_0,
}
sd_models = {ModelId.sd_v1_5}


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
    load_model = None

    @abstractmethod
    def diffuse_sample(self, sample: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def vae(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def image_processor(self):
        raise NotImplementedError


class SDPipe(DiffusionModel):
    def __init__(
        self,
        configs: dict[str, Any] = None,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()
        if configs is None:
            configs = {}

        base_model_id = configs.get("base_model_id", ModelId.sd_v1_5)
        refiner_model_id = configs.get("refiner_model_id", None)
        low_mem_mode = configs.get("low_mem_mode", False)
        compile_model = configs.get("compile_model", False)

        self.use_refiner = refiner_model_id is not None

        if self.use_refiner:
            raise NotImplementedError

        self.base_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base_pipe = prep_model(
            self.base_pipe,
            low_mem_mode=low_mem_mode,
            device=device,
            compile=compile_model,
        )
        self.base_pipe.safety_checker = None
        self.base_pipe.requires_safety_checker = False

        if self.use_refiner:
            self.refiner_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner_pipe = prep_model(
                self.refiner_pipe,
                low_mem_mode=low_mem_mode,
                device=device,
                compile=compile_model,
            )
        self.tokenizer = self.base_pipe.tokenizer
        self.text_encoder = self.base_pipe.text_encoder
        self.device = device

    def diffuse_sample(
        self,
        sample: dict[str, Any],
        base_strength: float = 0.2,
        refiner_strength: float = 0.2,
        base_denoising_start: int = None,
        base_denoising_end: int = None,
        refiner_denoising_start: int = None,
        refiner_denoising_end: int = None,
        original_size: tuple[int, int] = (1024, 1024),
        target_size: tuple[int, int] = (1024, 1024),
        prompt: str = default_prompt,
        negative_prompt: str = default_negative_prompt,
        base_num_steps: int = 50,
        refiner_num_steps: int = 50,
        base_gen: torch.Generator | Iterable[torch.Generator] = None,
        refiner_gen: torch.Generator | Iterable[torch.Generator] = None,
        base_kwargs: dict[str, any] = None,
        refiner_kwargs: dict[str, any] = None,
    ):
        image = sample["rgb"]
        batch_size = len(image)

        image = batch_if_not_iterable(image)
        base_gen = batch_if_not_iterable(base_gen)
        refiner_gen = batch_if_not_iterable(refiner_gen)
        validate_same_len(image, base_gen, refiner_gen)

        if base_gen:
            base_gen = base_gen * batch_size

        if refiner_gen:
            refiner_gen = refiner_gen * batch_size

        base_kwargs = base_kwargs or {}
        if prompt is not None:
            with torch.no_grad():
                tokens = tokenize_prompt(self.tokenizer, prompt).to(self.device)
                prompt_embeds = encode_tokens(
                    self.text_encoder, tokens, using_sdxl=False
                )["embeds"]
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)

        if negative_prompt is not None:
            with torch.no_grad():
                negative_tokens = tokenize_prompt(self.tokenizer, negative_prompt).to(
                    self.device
                )
                negative_prompt_embeds = encode_tokens(
                    self.text_encoder, negative_tokens, using_sdxl=False
                )["embeds"]

            negative_prompt_embeds = negative_prompt_embeds.expand(batch_size, -1, -1)
        image = self.base_pipe(
            image=image,
            generator=base_gen,
            output_type="latent" if self.use_refiner else "pt",
            strength=base_strength,
            denoising_start=base_denoising_start,
            denoising_end=base_denoising_end,
            num_inference_steps=base_num_steps,
            original_size=original_size,
            target_size=target_size,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            **base_kwargs,
        ).images

        if self.use_refiner:
            refiner_kwargs = refiner_kwargs or {}
            image = self.refiner_pipe(
                image=image,
                generator=refiner_gen,
                output_type="pt",
                strength=refiner_strength,
                denoising_start=refiner_denoising_start,
                denoising_end=refiner_denoising_end,
                num_inference_steps=refiner_num_steps,
                original_size=original_size,
                target_size=target_size,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **refiner_kwargs,
            ).images

        return {"rgb": image}

    @property
    def vae(self) -> AutoencoderKL:
        return self.base_pipe.vae

    @property
    def image_processor(self) -> VaeImageProcessor:
        return self.base_pipe.image_processor


model_name_to_constructor = {
    "sd_base": SDPipe,
    "sd_full": SDPipe,
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


def load_img2img_model(
    model_config_params: dict[str, Any], device=get_device()
) -> DiffusionModel:
    logging.info(f"Loading diffusion model...")

    model_name = model_config_params.get("model_name")
    constructor = model_name_to_constructor.get(model_name)
    if not constructor:
        raise NotImplementedError

    model = constructor(configs=model_config_params, device=device)
    logging.info(f"Finished loading diffusion model")
    return model


DiffusionModel.load_model = load_img2img_model


@lru_cache(maxsize=4)
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
