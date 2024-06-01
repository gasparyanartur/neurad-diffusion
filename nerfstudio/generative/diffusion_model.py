from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Union, Optional
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import logging
import itertools as it
import yaml

import numpy as np
import torch
from torch import Tensor, nn

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL


import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transform
from torchmetrics.image import PeakSignalNoiseRatio

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from nerfstudio.configs.base_config import InstantiateConfig


def get_device():
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


def _prepare_pipe(
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


def _make_metric(name, device, **kwargs):
    match name:
        case "psnr":
            metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

        case "mse":
            metric = nn.MSELoss().to(device)

        case _:
            raise NotImplementedError

    return metric


@dataclass
class DiffusionModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"
    mock = "mock"


@dataclass
class DiffusionModelType:
    sd: str = "sd"
    mock: str = "mock"


def tokenize_prompt(
    tokenizer: CLIPTokenizer, prompt: Union[str, Iterable[str]]
) -> torch.Tensor:
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = text_inputs.input_ids
    return tokens


def encode_tokens(
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    tokens: torch.Tensor,
) -> torch.Tensor:
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
) -> torch.Tensor:
    if not isinstance(
        prompt, str
    ):  # Convert list to tuple to make it hashable for memoization
        prompt = tuple(prompt)

    embeddings = _embed_hashable_prompt(tokenizer, text_encoder, prompt)
    return embeddings


@dataclass
class DiffusionModelConfig(InstantiateConfig):
    _target: DiffusionModel = field(default_factory=lambda: DiffusionModel.from_config)

    model_type: str = DiffusionModelType.sd
    model_id: str = DiffusionModelId.sd_v2_1

    low_mem_mode: bool = False
    """If applicable, prioritize options which lower GPU memory requirements at the expense of performance."""

    compile_model: bool = False
    """If applicable, compile Diffusion pipeline using available torch backend."""

    lora_weights: Optional[str] = None
    """Path to lora weights for the base diffusion model. Loads if applicable."""

    noise_strength: Optional[float] = 0.2
    """How much noise to apply during inference. 1.0 means complete gaussian."""

    num_inference_steps: Optional[int] = 50
    """Across how many timesteps the diffusion denoising occurs. Higher number gives better diffusion at expense of performance."""

    enable_progress_bar: bool = False
    """Create a progress bar for the denoising timesteps during inference."""

    metrics: Tuple[str, ...] = ("psnr", "mse")

    losses: Tuple[str, ...] = ("mse",)


class DiffusionModel(ABC):
    config: DiffusionModelConfig

    @abstractmethod
    def get_diffusion_output(
        self, sample: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: DiffusionModelConfig, **kwargs) -> "DiffusionModel":
        model_type_to_constructor = {
            DiffusionModelType.sd: StableDiffusionModel,
            DiffusionModelType.mock: MockDiffusionModel,
        }
        model = model_type_to_constructor[config.model_type]

        if config.compile_model and config.lora_weights:
            logging.warning(
                "Compiling the model currently leads to a bug when a LoRA is loaded, proceed with caution"
            )

        return model(config=config, **kwargs)


class MockDiffusionModel(DiffusionModel):
    def __init__(
        self, config: DiffusionModelConfig, device=get_device(), *args, **kwargs
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.config = config

        self.diffusion_metrics = {
            metric_name: _make_metric(metric_name, device)
            for metric_name in config.metrics
        }
        self.diffusion_losses = {
            loss_name: _make_metric(loss_name, device) for loss_name in config.losses
        }

    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        image = sample["rgb"]

        if len(image.shape) == 3:
            image = image[None, ...]

        return {"rgb": image}

    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        return {
            metric_name: metric(rgb_pred, rgb_gt)
            for metric_name, metric in self.diffusion_metrics.items()
        }

    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        loss_dict = {}
        for loss_name, loss in self.diffusion_losses.items():
            if loss_name in metrics_dict:
                loss_dict[loss_name] = metrics_dict[loss_name]
                continue

            loss_dict[loss_name] = loss(rgb_pred, rgb_gt)

        return loss_dict


class StableDiffusionModel(DiffusionModel):
    config: DiffusionModelConfig

    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()
        self.config = config

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        self.pipe.set_progress_bar_config(disable=not config.enable_progress_bar)

        if config.lora_weights:
            self.pipe.load_lora_weights(config.lora_weights)

        self.pipe = _prepare_pipe(
            self.pipe,
            low_mem_mode=config.low_mem_mode,
            device=device,
            compile=config.compile_model,
        )

        # The diffusion model sometimes detects NSFW content on the AD images for some reason
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        self.device = device

        self.diffusion_metrics = {
            metric_name: _make_metric(metric_name, device)
            for metric_name in config.metrics
        }
        self.diffusion_losses = {
            loss_name: _make_metric(loss_name, device) for loss_name in config.losses
        }

    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        **kwargs,
    ):
        """Denoise image with diffusion model.

        Interesting kwargs:
        - image
        - generator
        - output_type
        - strength
        - num_inference_steps
        - prompt_embeds
        - negative_prompt_embeds

        - denoising_start (sdxl)
        - denoising_end (sdxl)
        - original_size (sdxl)
        - target_size (sdxl)
        """
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
            image = image.permute(0, 3, 1, 2)  # Diffusion model is channel first

        kwargs = kwargs or {}
        kwargs["image"] = image
        kwargs["output_type"] = kwargs.get("output_type", "pt")
        kwargs["strength"] = kwargs.get("strength", self.config.noise_strength)
        kwargs["num_inference_steps"] = kwargs.get(
            "num_inference_steps", self.config.num_inference_steps
        )

        if (
            "generator" in kwargs
            and isinstance(kwargs["generator"], torch.Generator)
            and batch_size > 1
        ):
            raise ValueError(f"Number of generators must match number of images")

        # Convert any existing prompts to prompt embeddings, utilizing memoization.
        # Ensure there is at least one prompt embedding passed to the pipeline.
        prompt_embed_keys = []
        for prefix, suffix in it.product(["", "negative_"], ["", "_two"]):
            prompt_key = f"{prefix}prompt{suffix}"
            prompt_embed_key = f"{prefix}prompt_embeds{suffix}"

            if prompt_key in kwargs:
                prompt_embed_keys.append(prompt_embed_key)
                prompt = kwargs.pop(prompt_key)
                if prompt_embed_key not in kwargs:
                    with torch.no_grad():
                        kwargs[prompt_embed_key] = embed_prompt(
                            self.pipe.tokenizer, self.pipe.text_encoder, prompt
                        )

            if prompt_embed_key in kwargs:
                prompt_embed_keys.append(prompt_embed_key)

        # If no promp embed keys were passed, create one from an empty prompt
        if not prompt_embed_keys:
            prompt_embed_keys.append("prompt_embeds")
            with torch.no_grad():
                kwargs["prompt_embeds"] = embed_prompt(
                    self.pipe.tokenizer, self.pipe.text_encoder, ""
                )

        # Ensure batch size of prompts matches batch size of images
        for prompt_embed_key in prompt_embed_keys:
            if len(kwargs[prompt_embed_key].shape) == 2:
                kwargs[prompt_embed_key] = kwargs[prompt_embed_key][None, ...]

            if kwargs[prompt_embed_key].size(0) == 1 and batch_size > 1:
                embed_size = kwargs[prompt_embed_key].shape
                kwargs[prompt_embed_key] = kwargs[prompt_embed_key].expand(
                    batch_size * embed_size[0], embed_size[1], embed_size[2]
                )

        image = self.pipe(
            **kwargs,
        ).images

        if isinstance(image, list):
            image = torch.stack(image)

        if not channel_first:
            image = image.permute(0, 2, 3, 1)

        return {"rgb": image}

    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        return {
            metric_name: metric(rgb_pred, rgb_gt)
            for metric_name, metric in self.diffusion_metrics.items()
        }

    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        loss_dict = {}
        for loss_name, loss in self.diffusion_losses.items():
            if loss_name in metrics_dict:
                loss_dict[loss_name] = metrics_dict[loss_name]
                continue

            loss_dict[loss_name] = loss(rgb_pred, rgb_gt)

        return loss_dict
