from abc import ABC, abstractmethod
import warnings
from typing import Union, Optional, Tuple, Dict, Iterable, Any, List
from functools import lru_cache
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import itertools as it
import typing

import torch
from torch import FloatTensor, nn, Tensor

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.controlnet import (
    ControlNetModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor


import torchvision

from nerfstudio.configs.base_config import InstantiateConfig

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transform
from torchmetrics.image import PeakSignalNoiseRatio

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PretrainedConfig,
)
from transformers import AutoTokenizer

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from nerfstudio.generative.control_lora import ControlLoRAModel
from nerfstudio.generative.utils import (
    get_device,
    validate_same_len,
    batch_if_not_iterable,
)
from nerfstudio.generative.dynamic_dataset import (
    ConditioningSignalInfo,
    save_image,
    DynamicDataset,
    DATA_SUFFIXES,
)


default_prompt = ""
default_negative_prompt = ""

LOWER_DTYPES = {"fp16", "bf16"}
DTYPE_CONVERSION = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_metric(name, device, **kwargs):
    if name == "psnr":
        metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    elif name == "mse":
        metric = nn.MSELoss().to(device)

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
    hfsd: str = "hfsd"
    hfcn: str = "hfcn"
    sd: str = "sd"
    cn: str = "cn"
    mock: str = "mock"


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
    image = batch_if_not_iterable(image)
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
    sample: Dict[str, Tensor], conditioning_signal_infos: List["ConditioningSignalInfo"]
) -> torch.Tensor:
    signals = []

    for signal_info in conditioning_signal_infos:
        signal = sample[signal_info.name]
        signal: Tensor = batch_if_not_iterable(signal)

        if signal.size(1) != signal_info.num_channels:
            if signal.size(-1) != signal_info.num_channels:
                raise ValueError(
                    f"Invalid shape for conditioning signal: {signal_info}, received tensor with shape {signal.shape}"
                )

            signal = signal.permute(0, 2, 3, 1)

        signals.append(signal)

    return torch.cat(signals, dim=1)


def _prepare_conditioning(
    kwargs: Dict[str, Any],
    sample: Dict[str, Tensor],
    conditioning_signal_infos: List["ConditioningSignalInfo"],
) -> None:
    kwargs["control_image"] = combine_conditioning_info(
        sample, conditioning_signal_infos
    )


def _prepare_prompt(
    sample: Dict[str, Any],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    batch_size: int,
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
        if len(sample["generator"]) <= 1 and batch_size > 1:
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


@dataclass
class DiffusionModelConfig(InstantiateConfig):
    _target: "DiffusionModel" = field(
        default_factory=lambda: DiffusionModel.from_config
    )

    model_type: str = DiffusionModelType.sd
    model_id: str = DiffusionModelId.sd_v2_1

    low_mem_mode: bool = False
    """If applicable, prioritize options which lower GPU memory requirements at the expense of performance."""

    compile_model: bool = False
    """If applicable, compile Diffusion pipeline using available torch backend."""

    dtype: str = "fp32"
    """Data type of the underlying diffusion model. Options (fp32, fp16, bf16 (untested))"""

    lora_weights: Optional[str] = None
    """Path to lora weights for the base diffusion model. Loads if applicable."""

    controlnet_weights: Optional[str] = None
    """Path to lora weights for the base diffusion model. Loads if applicable."""

    noise_strength: float = 0.2
    """How much noise to apply during inference. 1.0 means complete gaussian."""

    num_inference_steps: int = 50
    """Across how many timesteps the diffusion denoising occurs. Higher number gives better diffusion at expense of performance."""

    enable_progress_bar: bool = False
    """Create a progress bar for the denoising timesteps during inference."""

    metrics: Tuple[str, ...] = ("psnr", "mse")

    losses: Tuple[str, ...] = ("mse",)

    conditioning_signals: Tuple[str, ...] = ()
    """ The name of the conditioning signals used for the controlnet.

        The signal should match the format `cn_{cn_type}_{num_channels}_{camera}`. 
            Eg. cn_rgb_3_front, cn_ray_6_front_left.
        During inference, the input must contain this signal as a key.

        Does nothing unless the model is a controlnet.
    """


class DiffusionModel(ABC):
    config: DiffusionModelConfig

    @abstractmethod
    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
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
    def from_config(
        cls,
        config: DiffusionModelConfig,
        pipe_models: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "DiffusionModel":
        pipe_models = pipe_models or {}
        model_type_to_constructor = {
            DiffusionModelType.hfsd: HFStableDiffusionModel,
            DiffusionModelType.hfcn: HFControlNetDiffusionModel,
            DiffusionModelType.sd: StableDiffusionModel,
            DiffusionModelType.cn: ControlNetDiffusionModel,
            DiffusionModelType.mock: MockDiffusionModel,
        }
        model = model_type_to_constructor[config.model_type]

        if config.compile_model and config.lora_weights:
            logging.warning(
                "Compiling the model currently leads to a bug when a LoRA is loaded, proceed with caution"
            )

        return model(config=config, pipe_models=pipe_models, **kwargs)


class MockDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionModelConfig,
        pipe_models: Dict[str, Any] = None,
        device=get_device(),
        *args,
        **kwargs,
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
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
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


class HFStableDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        pipe_models: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        pipe_models = pipe_models or {}

        self.config = config
        self.device = device

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=DTYPE_CONVERSION[config.dtype],
            variant=variant,
            use_safetensors=use_safetensors,
            **pipe_models,
        )

        self.pipe = prep_hf_pipe(
            self.pipe,
            device=device,
            low_mem_mode=config.low_mem_mode,
            compile=config.compile_model,
            num_inference_steps=config.num_inference_steps,
        )

        if (
            config.lora_weights
            and config.lora_weights != ""
            and config.lora_weights != "_"
        ):
            self.pipe.load_lora_weights(config.lora_weights)

        if verbose and kwargs:
            logging.info(f"Ignoring unrecognized kwargs: {kwargs.keys()}")

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
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
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
        pipeline_kwargs = pipeline_kwargs or {}
        pipeline_kwargs["image"] = sample["rgb"]
        pipeline_kwargs["output_type"] = pipeline_kwargs.get("output_type", "pt")
        pipeline_kwargs["strength"] = pipeline_kwargs.get(
            "strength", self.config.noise_strength
        )
        pipeline_kwargs["num_inference_steps"] = pipeline_kwargs.get(
            "num_inference_steps", self.config.num_inference_steps
        )

        channel_first, batch_size = _prepare_image(pipeline_kwargs)
        _prepare_generator(pipeline_kwargs, batch_size)
        _prepare_prompt(
            pipeline_kwargs, self.pipe.tokenizer, self.pipe.text_encoder, batch_size
        )

        image = self.pipe(
            **pipeline_kwargs,
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


class HFControlNetDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        revision: Optional[str] = None,
        pipe_models: dict[str, Any] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        pipe_models = pipe_models or {}

        self.config = config
        self.device = device

        self.conditioning_signal_infos = typing.cast(
            List[ConditioningSignalInfo],
            list(
                filter(
                    lambda s: s,
                    map(
                        ConditioningSignalInfo.from_signal_name,
                        self.config.conditioning_signals,
                    ),
                )
            ),
        )

        self.num_conditioning_channels = sum(
            signal.num_channels for signal in self.conditioning_signal_infos
        )

        pipe_models["unet"] = UNet2DConditionModel.from_pretrained(
            config.model_id,
            subfolder="unet",
            revision=revision,
            variant=variant,
            torch_dtype=DTYPE_CONVERSION[config.dtype],
            device=device,
            use_safetensors=use_safetensors,
        )

        pipe_models["vae"] = AutoencoderKL.from_pretrained(
            config.model_id,
            subfolder="vae",
            revision=revision,
            variant=variant,
            torch_dtype=DTYPE_CONVERSION[config.dtype],
            device=device,
            use_safetensors=use_safetensors,
        )

        if config.controlnet_weights and Path(config.controlnet_weights).exists():
            pipe_models["controlnet"] = ControlLoRAModel.from_pretrained(
                config.controlnet_weights,
                torch_dtype=DTYPE_CONVERSION[config.dtype],
                device=device,
                use_safetensors=use_safetensors,
            )
            pipe_models["controlnet"].tie_weights(pipe_models["unet"])
            pipe_models["controlnet"].bind_vae(pipe_models["vae"])

        else:
            logging.warning(
                f"Could not find controlnet weights. The pipeline will be loaded without them, which is the same as regular StableDiffusion. This behavior is unstable nad might crash during inference."
            )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=DTYPE_CONVERSION[config.dtype],
            variant=variant,
            use_safetensors=use_safetensors,
            **pipe_models,
        )

        self.pipe = prep_hf_pipe(
            self.pipe,
            low_mem_mode=config.low_mem_mode,
            device=device,
            compile=config.compile_model,
            num_inference_steps=config.num_inference_steps,
        )

        if config.lora_weights:
            self.pipe.load_lora_weights(config.lora_weights)

        if verbose and kwargs:
            logging.info(f"Ignoring unrecognized kwargs: {kwargs.keys()}")

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
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
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
        pipeline_kwargs = pipeline_kwargs or {}
        pipeline_kwargs["image"] = sample["rgb"]
        pipeline_kwargs["output_type"] = pipeline_kwargs.get("output_type", "pt")
        pipeline_kwargs["strength"] = pipeline_kwargs.get(
            "strength", self.config.noise_strength
        )
        pipeline_kwargs["num_inference_steps"] = pipeline_kwargs.get(
            "num_inference_steps", self.config.num_inference_steps
        )

        channel_first, batch_size = _prepare_image(pipeline_kwargs)

        _prepare_conditioning(pipeline_kwargs, sample, self.conditioning_signal_infos)
        _prepare_generator(pipeline_kwargs, batch_size)
        _prepare_prompt(
            pipeline_kwargs, self.pipe.tokenizer, self.pipe.text_encoder, batch_size
        )

        image = self.pipe(
            **pipeline_kwargs,
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


class StableDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        revision: Optional[str] = None,
        pipe_models: Optional[Dict[str, Any]] = None,
        requires_grad: bool = False,
        do_classifier_free_guidance: bool = True,
    ):
        super().__init__()

        pipe_models = pipe_models or {}

        self.config = config
        self.device = device

        dtype = DTYPE_CONVERSION[config.dtype]

        self.unet = typing.cast(
            UNet2DConditionModel,
            (
                pipe_models["unet"]
                if "unet" in pipe_models
                else UNet2DConditionModel.from_pretrained(
                    config.model_id,
                    subfolder="unet",
                    revision=revision,
                    variant=variant,
                    use_safetensors=use_safetensors,
                    torch_dtype=dtype,
                )
            ),
        )

        self.tokenizer = typing.cast(
            CLIPTokenizer,
            (
                pipe_models["tokenizer"]
                if "tokenizer" in pipe_models
                else AutoTokenizer.from_pretrained(
                    config.model_id,
                    subfolder="tokenizer",
                    revision=revision,
                    variant=variant,
                    use_fast=False,
                    use_safetensors=True,
                    torch_dtype=dtype,
                )
            ),
        )

        self.img_processor = (
            pipe_models["img_processor"]
            if "img_processor" in pipe_models
            else VaeImageProcessor()
        )

        self.noise_scheduler = typing.cast(
            DDPMScheduler,
            (
                pipe_models["noise_scheduler"]
                if "noise_scheduler" in pipe_models
                else DDPMScheduler.from_pretrained(
                    config.model_id,
                    subfolder="scheduler",
                    variant=variant,
                    revision=revision,
                    use_safetensors=use_safetensors,
                    torch_dtype=dtype,
                )
            ),
        )

        self.text_encoder = typing.cast(
            CLIPTextModel,
            (
                pipe_models["text_encoder"]
                if "text_encoder" in pipe_models
                else import_encoder(
                    config.model_id,
                    revision,
                    "text_encoder",
                    variant,
                    use_safetensors=use_safetensors,
                    torch_dtype=dtype,
                )
            ),
        )

        self.vae = (
            pipe_models["vae"]
            if "vae" in pipe_models
            else typing.cast(
                AutoencoderKL,
                AutoencoderKL.from_pretrained(
                    config.model_id,
                    subfolder="vae",
                    revision=revision,
                    variant=variant,
                    use_safetensors=use_safetensors,
                    torch_dtype=dtype,
                ),
            )
        )

        self.requires_grad = requires_grad

        self.vae.requires_grad_(requires_grad)
        self.vae.to(device=device, dtype=dtype)

        self.unet.requires_grad_(requires_grad)
        self.unet.to(device=device, dtype=dtype)

        self.text_encoder.requires_grad_(requires_grad)
        self.text_encoder.to(device=device, dtype=dtype)

        self.do_classifier_free_guidance = do_classifier_free_guidance

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
        strength: float = 0.2,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        if self.requires_grad:
            return self._get_diffusion_output(
                sample, strength, num_inference_steps, **kwargs
            )

        else:
            with torch.no_grad():
                return self._get_diffusion_output(
                    sample, strength, num_inference_steps, **kwargs
                )

    def _get_diffusion_output(
        self,
        sample: Dict[str, Any],
        strength: float = 0.2,
        num_inference_steps: int = 50,
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

        """

        if "generator" in kwargs:
            logging.warning(f"Provided generator, but not implementet yet.")

        image = sample["rgb"]
        image = batch_if_not_iterable(image)
        batch_size = len(image)

        if image.size(1) == 3:
            channel_first = True
        elif image.size(3) == 3:
            channel_first = False
        else:
            raise ValueError(f"Image needs to be BCHW or BHWC, received {image.shape}")

        if not channel_first:
            image = image.permute(0, 3, 1, 2)  # Diffusion model is channel first

        _prepare_prompt(sample, self.tokenizer, self.text_encoder, batch_size)

        latent = encode_img(
            self.img_processor,
            self.vae,
            image,
            device=self.device,
            seed=kwargs.get("seed"),
        )

        timesteps, num_inference_steps = get_timesteps(
            self.noise_scheduler, num_inference_steps, strength, self.device
        )
        noisy_latent = add_noise_to_latent(latent, timesteps[0], self.noise_scheduler)
        prompt_embeds = sample["prompt_embeds"]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat((prompt_embeds, prompt_embeds))

        denoised_latent = denoise_latent(
            noisy_latent, self.unet, timesteps, self.noise_scheduler, prompt_embeds
        )

        image = decode_img(self.img_processor, self.vae, denoised_latent)

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


class ControlNetDiffusionModel(DiffusionModel): ...


def encode_img(
    img_processor: VaeImageProcessor,
    vae: AutoencoderKL,
    img: Tensor,
    device: torch.device,
    seed: Optional[int] = None,
    sample_mode: str = "sample",
) -> Tensor:
    img = img_processor.preprocess(img)

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to(device))
    latents = retrieve_latents(
        latents,
        generator=(torch.manual_seed(seed) if seed is not None else None),
        sample_mode=sample_mode,
    )
    latents = latents * vae.config.scaling_factor

    if needs_upcasting:
        vae.to(original_vae_dtype)
        latents = latents.to(original_vae_dtype)

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
        generator=torch.manual_seed(seed) if (seed is not None) else None,
    )

    timesteps = torch.tensor([timestep], device=latent.device, dtype=torch.int)
    noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)
    return noisy_latents


# TODO: Denoise latent
def denoise_latent(
    latent: Tensor,
    unet: UNet2DConditionModel,
    timesteps: Union[List[int], Tensor],
    noise_scheduler: DDPMScheduler,
    encoder_hidden_states,
    timestep_cond=None,
    cross_attention_kwargs=None,
    added_cond_kwargs=None,
    extra_step_kwargs: Optional[Dict[str, Any]] = None,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = 7.5,
):
    if extra_step_kwargs is None:
        extra_step_kwargs = {}

    if timesteps[0] < timesteps[-1]:
        timesteps = reversed(timesteps)

    for t in timesteps:
        latent_model_input = (
            torch.cat([latent, latent]) if do_classifier_free_guidance else latent
        )
        # latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latent = noise_scheduler.step(
            noise_pred, t, latent, **extra_step_kwargs, return_dict=False
        )[0]

    return latent


def get_noised_img(img, timestep, vae, img_processor, noise_scheduler, seed=None):
    with torch.no_grad():
        model_input = encode_img(
            img_processor, vae, img, device=vae.device, seed=seed, sample_mode="sample"
        )
        noise = torch.randn_like(model_input, device=vae.device)
        timestep = noise_scheduler.timesteps[timestep]
        timesteps = torch.tensor([timestep], device=vae.device)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        img_pred = decode_img(img_processor, vae, noisy_model_input)

    return img_pred


@lru_cache(maxsize=4)
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
    buckets = torch.round(torch.linspace(start, end, n_draws + 1)).int()

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


def get_ordered_timesteps(
    noise_strength: float,
    device: Union[torch.device, str],
    total_num_timesteps: int = 1000,
    num_timesteps: Optional[int] = None,
    sample_from_bins: bool = True,
    low_noise_high_step: bool = False,
):
    warnings.warn(
        "Using deprecated `get_ordered_timesteps`, use `get_timesteps` instead"
    )
    if num_timesteps is None:
        num_timesteps = total_num_timesteps

    if low_noise_high_step:
        start_step = int((1 - noise_strength) * total_num_timesteps) - 1
        end_step = total_num_timesteps - 1
    else:
        start_step = 0
        end_step = int(noise_strength * total_num_timesteps)

    # start_step = int((1 - noise_strength) * total_num_timesteps)
    # end_step = total_num_timesteps - 1

    # Make sure the last one is total_num_timesteps-1
    if sample_from_bins:
        timesteps = draw_from_bins(
            start_step, end_step, num_timesteps - 1, include_last=True, device=device
        )
    else:
        timesteps = torch.round(
            torch.linspace(start_step, end_step, num_timesteps, device=device)
        ).to(torch.long)

    return timesteps


def get_matching(model, patterns: Iterable[Union[re.Pattern, str]] = (".*",)):
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, str):
            patterns[i] = re.compile(pattern)

    li = []
    for name, mod in model.named_modules():
        for pattern in patterns:
            if pattern.match(name):
                li.append((name, mod))
    return li


def parse_target_ranks(target_ranks, prefix=r""):
    parsed_targets = {}

    for name, item in target_ranks.items():
        if not item:
            continue

        match name:
            case "":
                continue

            case "downblocks":
                assert isinstance(item, dict)
                parsed_targets.update(
                    parse_target_ranks(item, rf"{prefix}.*down_blocks")
                )

            case "midblocks":
                assert isinstance(item, dict)
                parsed_targets.update(
                    parse_target_ranks(item, rf"{prefix}.*mid_blocks")
                )

            case "upblocks":
                assert isinstance(item, dict)
                parsed_targets.update(parse_target_ranks(item, rf"{prefix}.*up_blocks"))

            case "attn":
                assert isinstance(item, int)
                parsed_targets[f"{prefix}.*attn.*to_[kvq]"] = item
                parsed_targets[rf"{prefix}.*attn.*to_out\.0"] = item

            case "resnet":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*resnets.*conv\d*"] = item
                parsed_targets[rf"{prefix}.*resnets.*time_emb_proj"] = item

            case "ff":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*ff\.net\.0\.proj"] = item
                parsed_targets[rf"{prefix}.*ff\.net\.2"] = item

            case "proj":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*attentions.*proj_in"] = item
                parsed_targets[rf"{prefix}.*attentions.*proj_out"] = item

            case "_":
                raise NotImplementedError(f"Unrecognized target: {name}")

    return parsed_targets
