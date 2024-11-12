from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import Enum
from typing import Type, Union, Optional, Tuple, Dict, Iterable, Any, List, cast
from functools import lru_cache
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import itertools as it
import typing
from copy import deepcopy

import numpy as np
from pydantic import BaseModel, Field
import torch
from torch import FloatTensor, nn, Tensor

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.training_utils import cast_training_params

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import ControlNetModel

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput

from peft import LoraConfig, get_peft_model, PeftModel


import torchvision


torchvision.disable_beta_transforms_warning()
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PretrainedConfig,
)
from transformers import AutoTokenizer

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

from nerfstudio.generative.utils import (
    DTYPE_CONVERSION,
    get_device,
    batch_if_not_iterable,
)


class Metrics:
    psnr = "psnr"
    mse = "mse"
    ssim = "ssim"
    lpips = "lpips"


class MetricCompareDirections:
    lt = "<"
    gt = ">"


metric_improvement_direction = {
    Metrics.lpips: MetricCompareDirections.lt,
    Metrics.ssim: MetricCompareDirections.gt,
    Metrics.psnr: MetricCompareDirections.gt,
    Metrics.mse: MetricCompareDirections.lt,
}


class LoraRanks(BaseModel):
    unet: int = 4
    controlnet: int = 4
    text_encoder: int = 4

    def __getitem__(self, model_name: str) -> int:
        if not hasattr(self, model_name):
            raise ValueError(f"Model name {model_name} not found in LoraRanks.")

        return getattr(self, model_name)


def is_metric_improved(
    metric_name: str,
    original_value: Union[float, Tensor],
    new_value: Union[float, Tensor],
    true_on_eq: bool = False,
):
    improve_dir = metric_improvement_direction[metric_name]

    if improve_dir == MetricCompareDirections.lt:
        condition = original_value < new_value
    elif improve_dir == MetricCompareDirections.gt:
        condition = original_value > new_value
    else:
        raise ValueError(f"Improvement direction not found for metric {metric_name}")

    if true_on_eq:
        condition = (original_value == new_value) | condition

    return condition


def _make_metric(name: str, device: torch.device) -> nn.Module:
    if name == Metrics.psnr:
        metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    elif name == Metrics.mse:
        metric = nn.MSELoss().to(device)

    elif name == Metrics.ssim:
        metric = StructuralSimilarityIndexMeasure(
            data_range=(0.0, 1.0), reduction="none"
        ).to(device)

    elif name == Metrics.lpips:
        metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    else:
        raise NotImplementedError

    return metric


@dataclass
class DiffusionModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"


@dataclass
class DiffusionModelType:
    sd: str = "sd"
    cn: str = "cn"
    mock: str = "mock"


@dataclass
class ConditioningSignalInfo:
    signal_name: str
    num_channels: int


CONDITIONING_SIGNAL_INFO = {
    "ray": ConditioningSignalInfo(signal_name="ray", num_channels=6)
}


def get_signal_info(signal_name: str) -> ConditioningSignalInfo:
    return CONDITIONING_SIGNAL_INFO[signal_name]


def prep_hf_pipe(
    pipe: Union[
        StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
    ],
    device: torch.device = get_device(),
    low_mem_mode: bool = False,
    compile: bool = True,
    num_inference_steps: int = 50,
) -> Union[StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline]:
    if compile:
        try:
            pipe.unet = torch.compile(pipe.unet, fullgraph=True)
        except AttributeError:
            logging.warn(f"No unet found in Pipe. Skipping compiling")

    if low_mem_mode:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    return pipe


def _prepare_image(kwargs):
    image: Tensor = kwargs["image"]
    image = cast(Tensor, batch_if_not_iterable(image))
    batch_size = len(image)

    if image.size(1) == 3:
        channel_first = True
    elif image.size(3) == 3:
        channel_first = False
    else:
        raise ValueError(f"Image needs to be BCHW or BHWC, received {image.shape}")

    if not channel_first:
        image = image.permute(0, 3, 1, 2)  # Diffusion model is channel first

    kwargs["image"] = image
    return channel_first, batch_size


def combine_conditioning_info(
    sample: Dict[str, Any], conditioning_signals: Dict[str, ConditioningSignalInfo]
) -> Tensor:
    signals = []

    for signal_name in conditioning_signals.keys():
        signal = sample[signal_name]
        assert isinstance(signal, Tensor)

        signal = cast(Tensor, batch_if_not_iterable(signal))
        signals.append(signal)

    return torch.cat(signals, dim=1)


"""
def _prepare_conditioning(
    kwargs: Dict[str, Any],
    sample: Dict[str, Tensor],
    conditioning_signal_infos: List["ConditioningSignalInfo"],
) -> None:
    kwargs["control_image"] = combine_conditioning_info(
        sample, conditioning_signal_infos
    )
"""


def _prepare_prompt(
    sample: Dict[str, Any],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    batch_size: int,
    prompt_key: str = "prompt",
) -> None:
    # Convert any existing prompts to prompt embeddings, utilizing memoization.
    # Ensure there is at least one prompt embedding passed to the pipeline.
    prompt_embed_keys = []
    for prefix, suffix in it.product(["", "negative_"], ["", "_two"]):
        prompt_key = f"{prefix}prompt{suffix}"
        prompt_embed_key = f"{prefix}prompt_embeds{suffix}"

        if prompt_key in sample:
            prompt_embed_keys.append(prompt_embed_key)
            prompt = sample.pop(prompt_key)
            if prompt_embed_key not in sample:
                with torch.no_grad():
                    sample[prompt_embed_key] = embed_prompt(
                        tokenizer, text_encoder, prompt
                    )

        if prompt_embed_key in sample:
            prompt_embed_keys.append(prompt_embed_key)

    # If no promp embed keys were passed, create one from an empty prompt
    if not prompt_embed_keys:
        prompt_embed_key = "prompt_embeds"
        prompt_embed_keys.append(prompt_embed_key)
        with torch.no_grad():
            sample[prompt_embed_key] = embed_prompt(tokenizer, text_encoder, "")

    # Ensure batch size of prompts matches batch size of images
    for prompt_embed_key in prompt_embed_keys:
        sample[prompt_embed_key] = batch_if_not_iterable(
            sample[prompt_embed_key], single_dim=2
        )

        embed_size = sample[prompt_embed_key].shape
        if embed_size[0] == 1 and batch_size > 1:
            sample[prompt_embed_key] = sample[prompt_embed_key].expand(
                batch_size * embed_size[0], embed_size[1], embed_size[2]
            )


def _prepare_generator(sample: Dict["str", Any], batch_size: int):
    if "generator" in sample:
        sample["generator"] = batch_if_not_iterable(sample["generator"])
        bs = len(sample["generator"])  # type: ignore
        if bs <= 1 and batch_size > 1:
            raise ValueError(f"Number of generators must match number of images")


def generate_noise_pattern(
    n_clusters: int = 10,
    cluster_size_min: float = 2,
    cluster_size_max: float = 8,
    noise_strength: float = 0.2,
    pattern: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    img_h: Optional[int] = None,
    img_w: Optional[int] = None,
    use_monocolor: bool = False,
):
    if pattern is None:
        assert batch_size
        assert img_w
        assert img_h
        pattern = torch.zeros((batch_size, img_h, img_w, 3), dtype=torch.float32)
        original_device = pattern.device

    else:
        original_device = pattern.device
        pattern = torch.clone(pattern).to("cpu")
        pattern = typing.cast(Tensor, batch_if_not_iterable(pattern))
        batch_size, _, img_h, img_w = pattern.shape

    is_channel_first = pattern.size(-3) == 3 and pattern.size(-1)
    if is_channel_first:
        pattern = pattern.permute(0, 2, 3, 1)

    size_min = torch.tensor([cluster_size_min]).expand(batch_size, n_clusters)
    min_to_max = cluster_size_max - cluster_size_min
    szs = size_min + min_to_max * torch.rand(batch_size, n_clusters)

    yc = torch.randint(img_h, size=(batch_size, n_clusters))
    xc = torch.randint(img_w, size=(batch_size, n_clusters))

    ymin = torch.clamp_min(yc - szs, 0)
    ymax = torch.clamp_max(yc + szs, img_h)
    xmin = torch.clamp_min(xc - szs, 0)
    xmax = torch.clamp_max(xc + szs, img_w)

    for b in range(batch_size):
        for c in range(n_clusters):
            x1 = int(xmin[b, c])
            x2 = int(xmax[b, c]) + 1
            y1 = int(ymin[b, c])
            y2 = int(ymax[b, c]) + 1

            if use_monocolor:
                noise_pattern = noise_strength * torch.rand([1, 1, 3])

            else:
                noise_pattern = noise_strength * torch.rand_like(
                    pattern[b, y1:y2, x1:x2]
                )

            pattern[b, y1:y2, x1:x2] += noise_pattern

    pattern[pattern < 0] = 0
    pattern[pattern > 1] = 1

    if is_channel_first:
        pattern = pattern.permute(0, 3, 1, 2)

    return pattern.to(device=original_device)


class DiffusionModelConfig(BaseModel):

    type: str = DiffusionModelType.sd
    id: str = DiffusionModelId.sd_v2_1
    variant: Optional[str] = None
    revision: str = "main"
    use_safetensors: bool = True

    compile_models: Tuple[str, ...] = ()
    """If applicable, compile Diffusion pipeline using available torch backend."""

    dtype: str = "fp32"
    """Data type of the underlying diffusion model. Options (fp32, fp16, bf16 (untested))"""

    lora_weights: Optional[str] = None
    """Path to the directory which contains saved model. Loads if applicable."""

    models_to_load_lora: Tuple[str, ...] = ()

    models_to_train_lora: Tuple[str, ...] = ()

    noise_strength: float = 0.1

    """How much noise to apply during inference. 1.0 means complete gaussian."""

    num_inference_steps: int = 50
    """Across how many timesteps the diffusion denoising occurs. Higher number gives better diffusion at expense of performance."""

    enable_progress_bar: bool = False
    """Create a progress bar for the denoising timesteps during inference."""

    metrics: Tuple[str, ...] = ("lpips", "ssim", "psnr", "mse")

    losses: Tuple[str, ...] = ("mse",)

    lora_ranks: LoraRanks = LoraRanks()

    train_attn_blocks: bool = True
    train_resnet_blocks: bool = True

    lora_modules_to_save: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "unet": (),
            "controlnet": tuple(
                [
                    "controlnet_cond_embedding",
                    *[f"controlnet_down_blocks.{i}" for i in range(12)],
                    "controlnet_mid_block",
                ]
            ),
        }
    )

    use_dora: bool = True

    lora_model_prefix: str = "lora_"

    # conditioning_signals: Dict[str, ConditioningSignalInfo] = field(
    #    default_factory=lambda: {
    #        "ray": ConditioningSignalInfo(signal_name="ray", num_channels=6)
    #    }
    # )

    conditioning_signals: Tuple[str, ...] = ("ray",)

    do_classifier_free_guidance: bool = True
    guidance_scale: float = 0
    conditioning_scale: float = 0.8

    def setup(self, **kwargs):
        return StableDiffusionModel(self, **kwargs)


def import_encoder_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: Optional[str],
    subfolder: Optional[str],
):
    if revision is None:
        revision = "main"

    encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        resume_download=False,
    )
    model_class = encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection

    elif model_class == "CLIPVisionModel":
        from transformers import CLIPVisionModel

        return CLIPVisionModel

    elif model_class == "CLIPVisionModelWithProjection":

        from transformers import CLIPVisionModelWithProjection

        return CLIPVisionModelWithProjection

    else:
        raise ValueError(f"{model_class} is not supported.")


def import_encoder(model_id, revision, subfolder, variant, **kwargs):
    return import_encoder_class_from_model_name_or_path(
        model_id, revision, subfolder=subfolder
    ).from_pretrained(
        model_id, subfolder=subfolder, revision=revision, variant=variant, **kwargs
    )


class StableDiffusionModel:
    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
        models: Optional[Dict[str, Any]] = None,
    ):
        models = models or {}

        self.config = config
        self.device = device

        dtype = DTYPE_CONVERSION[config.dtype]

        self.models = models

        if "unet" not in self.models:
            models["unet"] = UNet2DConditionModel.from_pretrained(
                config.id,
                subfolder="unet",
                revision=config.revision,
                variant=config.variant,
                use_safetensors=config.use_safetensors,
                torch_dtype=dtype,
            )

        if "tokenizer" not in self.models:
            models["tokenizer"] = AutoTokenizer.from_pretrained(
                config.id,
                subfolder="tokenizer",
                revision=config.revision,
                variant=config.variant,
                use_fast=False,
                use_safetensors=config.use_safetensors,
                torch_dtype=dtype,
            )

        if "img_processor" not in models:
            models["img_processor"] = VaeImageProcessor()

        if "noise_scheduler" not in models:
            models["noise_scheduler"] = DDPMScheduler.from_pretrained(
                config.id,
                subfolder="scheduler",
                variant=config.variant,
                revision=config.revision,
                use_safetensors=config.use_safetensors,
                torch_dtype=dtype,
            )

        if "text_encoder" not in models:
            models["text_encoder"] = import_encoder(
                config.id,
                config.revision,
                "text_encoder",
                config.variant,
                use_safetensors=config.use_safetensors,
                torch_dtype=dtype,
            )

        if "vae" not in models:
            models["vae"] = AutoencoderKL.from_pretrained(
                config.id,
                subfolder="vae",
                revision=config.revision,
                variant=config.variant,
                use_safetensors=config.use_safetensors,
                torch_dtype=dtype,
            )

        self.vae.requires_grad_(False)
        self.vae.to(device=device, dtype=dtype)

        self.unet.requires_grad_(False)
        self.unet.to(device=device, dtype=dtype)

        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(device=device, dtype=dtype)  # type: ignore

        if "controlnet" not in models:
            models["controlnet"] = ControlNetModel.from_unet(
                unet=self.unet,
                conditioning_channels=self.conditioning_channels,
            )

        self.controlnet.requires_grad_(False)
        self.controlnet.to(device=device, dtype=dtype)

        self.diffusion_metrics = {
            metric_name: _make_metric(metric_name, device)
            for metric_name in config.metrics
        }
        self.diffusion_losses = {
            loss_name: _make_metric(loss_name, device) for loss_name in config.losses
        }

        if self.config.models_to_load_lora:
            for model in self.config.models_to_load_lora:
                self.load_adapter(model, copy_model=True)

        if self.config.models_to_train_lora:
            for model in self.config.models_to_train_lora:
                self.add_adapter(model, copy_model=True)

            cast_training_params(
                [
                    self.models[model_name]
                    for model_name in self.config.models_to_train_lora
                ],
                dtype=torch.float32,
            )

        if self.config.compile_models:
            for model in self.config.compile_models:
                self.models[model] = torch.compile(self.models[model])

    @property
    def conditioning_signals(self) -> Dict[str, ConditioningSignalInfo]:
        return {
            signal_name: get_signal_info(signal_name)
            for signal_name in self.config.conditioning_signals
        }

    @property
    def conditioning_channels(self) -> int:
        return sum(signal.num_channels for signal in self.conditioning_signals.values())

    @property
    def using_controlnet(self) -> bool:
        return self.config.type == "cn"

    @property
    def controlnet(self) -> ControlNetModel:
        return self.models["controlnet"]

    @property
    def unet(self) -> UNet2DConditionModel:
        return self.models["unet"]

    @property
    def vae(self) -> AutoencoderKL:
        return self.models["vae"]

    @property
    def text_encoder(self) -> CLIPTextModel:
        return self.models["text_encoder"]

    @property
    def noise_scheduler(self) -> DDPMScheduler:
        return self.models["noise_scheduler"]

    @property
    def tokenizer(self) -> CLIPTokenizer:
        return self.models["tokenizer"]

    @property
    def img_processor(self) -> VaeImageProcessor:
        return self.models["img_processor"]

    @property
    def dtype(self) -> torch.dtype:
        return DTYPE_CONVERSION[self.config.dtype]

    def get_models_to_train(self) -> Iterable[PeftModel]:
        for model_name in self.config.models_to_train_lora:
            yield self.models[model_name]

    def get_models_to_train_with_name(self) -> Iterable[Tuple[str, PeftModel]]:
        for model_name in self.config.models_to_train_lora:
            yield model_name, self.models[model_name]

    def set_model(self, model_name: str, model: PeftModel) -> None:
        self.models[model_name] = model

    def set_models(self, model_names: Iterable[str], models: List[PeftModel]) -> None:
        for model_name, model in zip(model_names, models):
            self.set_model(model_name, model)

    def set_gradient_checkpointing(self, mode: bool = False) -> None:
        if mode:
            for model_name in self.config.models_to_train_lora:
                self.models[model_name].enable_gradient_checkpointing()
        else:
            for model_name in self.config.models_to_train_lora:
                self.models[model_name].disable_gradient_checkpointing()

    def set_training(self, mode: bool = False) -> None:
        for model_name in self.config.models_to_train_lora:
            self.models[model_name].train(mode)

    def is_model_trained(self, model_name: str) -> bool:
        return model_name in self.config.models_to_train_lora

    def load_adapter(self, model_name: str, copy_model: bool = True) -> None:
        # TODO: Test this

        if not self.config.lora_weights:
            raise ValueError(
                f"Cannot load adapter for {model_name} - not specify a path in `config.lora_weights`."
            )

        if not model_name in self.config.models_to_load_lora:
            raise ValueError(
                f"Cannot load adapter for {model_name} - model not in list of lora models: {self.config.models_to_load_lora}."
            )

        lora_path = Path(self.config.lora_weights)
        model_path = lora_path / (self.config.lora_model_prefix + model_name)

        model = self.models[model_name]

        if copy_model:
            model = deepcopy(model)

        model = PeftModel.from_pretrained(model, model_path)
        self.models[model_name] = model

    def add_adapter(
        self,
        model_name: str,
        adapter_config: Optional[LoraConfig] = None,
        copy_model: bool = True,
    ) -> None:

        if not model_name in self.config.models_to_train_lora:
            raise ValueError(
                f"Cannot add adapter for {model_name} - model not in list of trainable models: {self.config.models_to_train_lora}"
            )

        model = self.models[model_name]
        if adapter_config is None:
            model_target_ranks = {
                "downblocks": {
                    "attn": int(self.config.train_attn_blocks),
                    "resnet": int(self.config.train_resnet_blocks),
                },
                "midblocks": {
                    "attn": int(self.config.train_attn_blocks),
                    "resnet": int(self.config.train_resnet_blocks),
                },
                "upblocks": {
                    "attn": int(self.config.train_attn_blocks),
                    "resnet": int(self.config.train_resnet_blocks),
                },
            }

            model_ranks = parse_target_ranks(model_target_ranks)

            base_rank = self.config.lora_ranks[model_name]
            target_ranks = {k: v * base_rank for k, v in model_ranks.items()}

            modules_to_save = self.config.lora_modules_to_save[model_name]
            adapter_config = LoraConfig(
                r=base_rank,
                lora_alpha=base_rank,
                init_lora_weights="gaussian",
                use_dora=self.config.use_dora,
                rank_pattern=target_ranks,
                target_modules="|".join(target_ranks.keys()),
                modules_to_save=list(modules_to_save),
            )

        if copy_model:
            model = deepcopy(model)

        model = get_peft_model(
            model, adapter_config, self.config.lora_model_prefix + model_name
        )
        self.models[model_name] = model

    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        rgb_key: str = "rgb",
        nan_to_zero: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Denoise image with diffusion model.

        Interesting kwargs:
        - image
        - generator
        - output_type
        - strength
        - num_inference_steps
        - prompt_embeds
        - negative_prompt_embeds

        """

        if "generator" in kwargs:
            logging.warning(f"Provided generator, but not implementet yet.")

        image = sample[rgb_key]
        image = cast(Tensor, batch_if_not_iterable(image))

        image = image.to(device=self.device, dtype=self.dtype)

        input_ids = (
            sample["input_ids"]
            if "input_ids" in sample
            else tokenize_prompt(self.tokenizer, "")
        )

        strength = kwargs.get("strength", self.config.noise_strength)

        if image.size(1) == 3:
            channel_first = True
        elif image.size(3) == 3:
            channel_first = False
        else:
            raise ValueError(f"Image needs to be BCHW or BHWC, received {image.shape}")

        if not channel_first:
            image = image.permute(0, 3, 1, 2)  # Diffusion model is channel first

        with torch.no_grad():
            latent = encode_img(
                self.img_processor,
                self.vae,
                image,
                device=self.device,
                seed=kwargs.get("seed"),
            )

            timesteps, _ = get_timesteps(
                self.noise_scheduler,
                self.config.num_inference_steps,
                strength,
                self.device,
            )
            noisy_latent = add_noise_to_latent(
                latent, timesteps[0], self.noise_scheduler, seed=kwargs.get("seed")
            )

            prompt_embeds = encode_tokens(self.text_encoder, input_ids, True)

            if self.config.do_classifier_free_guidance:
                prompt_embeds = torch.cat((prompt_embeds, prompt_embeds))

            if self.using_controlnet:
                conditioning = combine_conditioning_info(
                    sample, self.conditioning_signals
                ).to(noisy_latent.device)

                if self.config.do_classifier_free_guidance:
                    conditioning = torch.cat((conditioning, conditioning))

            else:
                conditioning = None

            denoised_latent = denoise_latent(
                noisy_latent,
                self.unet,
                timesteps,  # type: ignore
                self.noise_scheduler,
                prompt_embeds,
                do_classifier_free_guidance=self.config.do_classifier_free_guidance,
                guidance_scale=self.config.guidance_scale,
                controlnet=self.controlnet if self.using_controlnet else None,
                controlnet_conditioning=conditioning,
            )

            image = decode_img(self.img_processor, self.vae, denoised_latent)

            if not channel_first:
                image = image.permute(0, 2, 3, 1)

        return {"rgb": image}

    def get_diffusion_metrics(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        pred_rgb_key="rgb",
        gt_rgb_key="rgb",
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred[pred_rgb_key]
        rgb_gt = batch_gt[gt_rgb_key]

        return {
            metric_name: metric(rgb_pred, rgb_gt)
            for metric_name, metric in self.diffusion_metrics.items()
        }

    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        pred_rgb_key="rgb",
        gt_rgb_key="rgb",
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred[pred_rgb_key]
        rgb_gt = batch_gt[gt_rgb_key]

        loss_dict = {}
        for loss_name, loss in self.diffusion_losses.items():
            loss_dict[loss_name] = loss(rgb_pred, rgb_gt)

        return loss_dict


def retrieve_latents(
    encoder_output: AutoencoderKLOutput,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
) -> Tensor:
    if sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def encode_img(
    img_processor: VaeImageProcessor,
    vae: AutoencoderKL,
    img: Tensor,
    device: torch.device,
    seed: Optional[int] = None,
    sample_mode: str = "sample",
) -> Tensor:
    img = img_processor.preprocess(img)  # type: ignore

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast  # type: ignore
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to(device), return_dict=True)  # type: ignore
    latents = retrieve_latents(
        latents,  # type: ignore
        generator=(torch.manual_seed(seed) if seed is not None else None),
        sample_mode=sample_mode,
    )
    latents = latents * vae.config.scaling_factor  # type: ignore

    if needs_upcasting:
        vae.to(original_vae_dtype)
        latents = latents.to(original_vae_dtype)

    return latents


def decode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, latents: Tensor
) -> Tensor:
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast  # type: ignore
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    img = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]  # type: ignore

    if needs_upcasting:
        vae.to(original_vae_dtype)

    img = img_processor.postprocess(img, output_type="pt")  # type: ignore
    return img  # type: ignore


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    vae.post_quant_conv.to(dtype)
    vae.decoder.conv_in.to(dtype)
    vae.decoder.mid_block.to(dtype)

    return vae


def add_noise_to_latent(
    latent: Tensor,
    timestep: Union[int, List[int], torch.Tensor],
    noise_scheduler: DDPMScheduler,
    seed: Optional[int] = None,
):
    if isinstance(timestep, int):
        timestep = [timestep]

    if isinstance(timestep, list):
        timestep = torch.tensor(timestep)

    noise = torch.randn(
        size=latent.shape,
        device=latent.device,
        dtype=latent.dtype,
        generator=(
            torch.Generator(device=latent.device).manual_seed(seed)
            if (seed is not None)
            else None
        ),
    )

    timesteps = torch.tensor([timestep], device=latent.device, dtype=torch.int)
    noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)  # type: ignore
    return noisy_latents


# TODO: Denoise latent
def denoise_latent(
    latent: Tensor,
    unet: UNet2DConditionModel,
    timesteps: Sequence[int],
    noise_scheduler: DDPMScheduler,
    encoder_hidden_states,
    controlnet: Optional[ControlNetModel] = None,
    controlnet_conditioning: Optional[Tensor] = None,
    timestep_cond=None,
    cross_attention_kwargs=None,
    added_cond_kwargs=None,
    extra_step_kwargs: Optional[Dict[str, Any]] = None,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = 0,
):
    if extra_step_kwargs is None:
        extra_step_kwargs = {}

    if timesteps[0] < timesteps[-1]:
        timesteps = reversed(timesteps)  # type: ignore

    for t in timesteps:
        latent_model_input = (
            torch.cat([latent, latent]) if do_classifier_free_guidance else latent
        )
        latent_model_input = noise_scheduler.scale_model_input(
            cast(FloatTensor, latent_model_input), t
        )

        unet_kwargs = {}
        if controlnet is not None:
            down_block_res_samples, mid_block_res_sample = controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_conditioning,
                return_dict=False,
            )

            unet_kwargs["down_block_additional_residuals"] = [
                sample.to(dtype=controlnet.dtype) for sample in down_block_res_samples
            ]
            unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample.to(
                dtype=controlnet.dtype
            )

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            **unet_kwargs,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latent = noise_scheduler.step(
            noise_pred, t, latent, **extra_step_kwargs, return_dict=False  # type: ignore
        )[0]

    return latent


def get_noised_img(
    img: Tensor,
    timestep: Union[int, Tensor],
    vae: AutoencoderKL,
    img_processor: VaeImageProcessor,
    noise_scheduler: DDPMScheduler,
    seed: Optional[int] = None,
) -> Tensor:
    with torch.no_grad():
        model_input = encode_img(
            img_processor, vae, img, device=vae.device, seed=seed, sample_mode="sample"
        )
        noise = torch.randn_like(model_input, device=vae.device)
        timesteps = torch.tensor(
            [noise_scheduler.timesteps[timestep]], device=vae.device
        )
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)  # type: ignore
        img_pred = decode_img(img_processor, vae, noisy_model_input)

    return img_pred


@lru_cache(maxsize=4)
def tokenize_prompt(
    tokenizer: CLIPTokenizer, prompt: Union[str, Iterable[str]]
) -> torch.Tensor:
    text_inputs = tokenizer(
        prompt,  # type: ignore
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = text_inputs.input_ids
    return tokens


@lru_cache(maxsize=4)
def _encode_hashable_tokens(
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    tokens: torch.Tensor,
) -> torch.Tensor:
    prompt_embeds = text_encoder(tokens.to(text_encoder.device))
    return prompt_embeds.last_hidden_state


def encode_tokens(
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    tokens: torch.Tensor,
    use_cache: bool = True,
) -> torch.Tensor:
    if use_cache:
        return _encode_hashable_tokens(text_encoder, tokens)

    prompt_embeds = text_encoder(tokens.to(text_encoder.device))
    return prompt_embeds.last_hidden_state


@lru_cache(maxsize=4)
def _embed_hashable_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    prompt: Union[str, Tuple[str, ...]],
) -> torch.Tensor:
    tokens = tokenize_prompt(tokenizer, prompt)
    embeddings = encode_tokens(text_encoder, tokens)
    return embeddings


def embed_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    prompt: Union[str, Tuple[str, ...]],
    use_cache: bool = True,
) -> torch.Tensor:
    if not isinstance(
        prompt, str
    ):  # Convert list to tuple to make it hashable for memoization
        prompt = tuple(prompt)

    if use_cache:
        embeddings = _embed_hashable_prompt(tokenizer, text_encoder, prompt)
    else:
        tokens = tokenize_prompt(tokenizer, prompt)
        embeddings = encode_tokens(text_encoder, tokens)

    return embeddings


def draw_from_bins(start, end, n_draws, device, include_last: bool = False):
    values = torch.zeros(n_draws + int(include_last), dtype=torch.long, device=device)
    buckets = np.round(np.linspace(start, end, n_draws + 1)).astype(int)

    for i in range(n_draws):
        values[i] = torch.randint(buckets[i], buckets[i + 1], (1,))

    if include_last:
        values[-1] = end

    return values


def get_random_timesteps(
    noise_strength,
    total_num_timesteps,
    device,
    batch_size,
    low_noise_high_step: bool = False,
):
    if low_noise_high_step:
        start_step = int((1 - noise_strength) * total_num_timesteps)
        end_step = total_num_timesteps
    else:
        start_step = 0
        end_step = int(noise_strength * total_num_timesteps)

    # Sample a random timestep for each image
    timesteps = torch.randint(
        start_step,
        end_step,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    return timesteps


def get_timesteps(
    noise_scheduler: DDPMScheduler,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
):
    # Adapted from diffusers.StableDiffusionImg2ImgPipeline.get_timesteps

    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start * noise_scheduler.order :]
    if hasattr(noise_scheduler, "set_begin_index"):
        noise_scheduler.set_begin_index(t_start * noise_scheduler.order)

    return timesteps, num_inference_steps - t_start


def get_matching(
    model: nn.Module, patterns: Iterable[Union[re.Pattern, str]] = (".*",)
):
    parsed_patterns = [
        re.compile(pattern) if isinstance(pattern, str) else pattern
        for pattern in patterns
    ]

    li = [
        (name, mod)
        for (name, mod), pattern in it.product(model.named_modules(), parsed_patterns)
        if pattern.match(name)
    ]

    return li


def parse_target_ranks(target_ranks, prefix=r""):
    parsed_targets = {}

    for name, item in target_ranks.items():
        if not item:
            continue

        if name == "":
            continue

        elif name == "downblocks":
            assert isinstance(item, dict)
            parsed_targets.update(parse_target_ranks(item, rf"{prefix}.*down_blocks"))

        elif name == "midblocks":
            assert isinstance(item, dict)
            parsed_targets.update(parse_target_ranks(item, rf"{prefix}.*mid_blocks"))

        elif name == "upblocks":
            assert isinstance(item, dict)
            parsed_targets.update(parse_target_ranks(item, rf"{prefix}.*up_blocks"))

        elif name == "attn":
            assert isinstance(item, int)
            parsed_targets[rf"{prefix}.*attn.*to_[kvq]"] = item
            parsed_targets[rf"{prefix}.*attn.*to_out\.0"] = item

        elif name == "resnet":
            assert isinstance(item, int)
            parsed_targets[rf"{prefix}.*resnets.*conv\d*"] = item
            parsed_targets[rf"{prefix}.*resnets.*time_emb_proj"] = item

        elif name == "ff":
            assert isinstance(item, int)
            parsed_targets[rf"{prefix}.*ff\.net\.0\.proj"] = item
            parsed_targets[rf"{prefix}.*ff\.net\.2"] = item

        elif name == "proj":
            assert isinstance(item, int)
            parsed_targets[rf"{prefix}.*attentions.*proj_in"] = item
            parsed_targets[rf"{prefix}.*attentions.*proj_out"] = item

        else:
            raise NotImplementedError(f"Unrecognized target: {name}")

    return parsed_targets
