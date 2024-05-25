from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Union, Optional
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import logging

import torch
from torch import Tensor, nn

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL

import numpy as np
import yaml
import torchvision

from nerfstudio.configs.base_config import InstantiateConfig

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transform
from torchmetrics.image import PeakSignalNoiseRatio


default_prompt = "dashcam recording, urban driving scene, video, autonomous driving, detailed cars, traffic scene, pandaset, kitti, high resolution, realistic, detailed, camera video, dslr, ultra quality, sharp focus, crystal clear, 8K UHD, 10 Hz capture frequency 1/2.7 CMOS sensor, 1920x1080"
default_negative_prompt = "face, human features, unrealistic, artifacts, blurry, noisy image, NeRF, oil-painting, art, drawing, poor geometry, oversaturated, undersaturated, distorted, bad image, bad photo"


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
            not "prompt" in kwargs
            and not "negative_prompt" in kwargs
            and not "prompt_embeds" in kwargs
            and not "negative_prompt_embeds" in kwargs
        ):
            kwargs["prompt"] = ""

        # Repeat the items which need to be given as a batch
        batch_size = len(image)
        for key in ["prompt", "negative_prompt", "prompt_embeds", "negative_prompt_embeds", "generator"]:
            if key not in kwargs:
                continue

            value = kwargs[key]
            if isinstance(value, (str, torch.Generator)):
                kwargs[key] = [value for _ in range(batch_size)]
            elif isinstance(value, torch.Tensor):
                kwargs[key] = [value for _ in range]



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
