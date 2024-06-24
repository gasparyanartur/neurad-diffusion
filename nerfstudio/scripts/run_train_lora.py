# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

from typing import Any, Callable, Dict, Tuple, List, Optional, Union, Sequence
from collections.abc import Iterable
from pathlib import Path
from argparse import ArgumentParser, Namespace
import logging
import os
from dataclasses import dataclass, asdict, field
import math
import itertools as it

import torch.utils
import torch.utils.data
import tqdm
import shutil
import functools

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import transformers
import torchvision.transforms.v2 as transforms
from transformers import AutoTokenizer
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.generative.diffusion_model import (
    import_encoder_class_from_model_name_or_path,
)
from nerfstudio.generative.dynamic_dataset import (
    INFO_GETTER_BUILDERS,
    CameraDataGetter,
    ConditioningSignalInfo,
    IntrinsicsDataGetter,
    PoseDataGetter,
    SampleInfo,
    TimestampDataGetter,
    crop_to_ray_idxs,
    read_data_tree,
    save_yaml,
    setup_project,
    DynamicDataset,
)
from nerfstudio.generative.diffusion_model import (
    generate_noise_pattern,
    get_noised_img,
    tokenize_prompt,
    encode_tokens,
    HFStableDiffusionModel,
    DiffusionModelConfig,
    DiffusionModel,
    decode_img,
    combine_conditioning_info,
    parse_target_ranks,
    get_random_timesteps,
    LOWER_DTYPES,
    DTYPE_CONVERSION,
    DiffusionModelType,
    DiffusionModelId,
)
from nerfstudio.generative.utils import get_env, nearest_multiple
from nerfstudio.generative.control_lora import ControlLoRAModel
from nerfstudio.model_components.ray_generators import RayGenerator

check_min_version("0.27.0")
logger = get_logger(__name__, log_level="INFO")


@dataclass(init=True)
class TrainState:
    job_id: Optional[str] = None
    project_name: str = "ImagineDriving"
    project_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    datasets: Dict[str, Any] = field(default_factory=dict)

    model_type: str = DiffusionModelType.sd
    model_id: str = DiffusionModelId.sd_v2_1

    # Path to pretrained VAE model with better numerical stability.
    # More details: https://github.com/huggingface/diffusers/pull/4038.
    # vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_id: Optional[str] = None

    weights_dtype: str = "fp32"
    vae_dtype: Optional[str] = "fp32"

    revision: Optional[str] = "main"
    variant: Optional[str] = None
    prediction_type: Optional[str] = None

    enable_gradient_checkpointing: bool = False

    checkpoint_strategy: str = "best"
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = (
        0  # How many checkpoints, besides the latest, that will be stored
    )
    resume_from_checkpoint: bool = False  # Does not work atm

    n_epochs: int = 100
    max_train_samples: Optional[int] = None
    val_freq: int = 10
    frac_dataset_per_epoch: float = 1.0

    train_noise_strength: float = 0.5
    train_noise_num_steps: Optional[int] = None
    val_noise_num_steps: int = 30
    val_noise_strength: float = 0.25
    use_cached_tokens: bool = True

    keep_vae_full: bool = False

    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
    noise_offset: float = 0

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = True
    torch_backend: str = "cudagraphs"

    scale_lr: bool = False
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    snr_gamma: Optional[float] = None
    max_grad_norm: float = 1.0

    lora_rank_linear: int = 4
    lora_rank_conv2d: int = 4
    use_dora: bool = False

    control_lora_rank_linear: int = 4
    control_lora_rank_conv2d: int = 4
    lora_target_ranks: Dict[str, Any] = field(
        default_factory=lambda: {
            "unet": {
                "downblocks": {"attn": 8, "resnet": 8, "ff": 4, "proj": 4},
                "midblocks": {"attn": 8, "resnet": 8, "ff": 4, "proj": 4},
                "upblocks": {"attn": 8, "resnet": 8, "ff": 4, "proj": 4},
            },
            "controlnet": {
                "downblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
                "midblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
            },  # ControlNet not implemented atm, refer to `control_lora_rank_linear` and `control_lora_rank_conv2d`
        }
    )
    lora_peft_type: str = (  # CURRENTLY NOT IN USE
        "LORA"  # LORA, LOHA, or LOKR. LOKR seems best for this task https://arxiv.org/pdf/2309.14859
    )

    conditioning_signals: List[str] = field(default_factory=lambda: [])
    conditioning_signal_infos: List[ConditioningSignalInfo] = field(
        default_factory=lambda: []
    )

    rec_loss_strength: float = 0.1

    max_train_steps: Optional[int] = None  # Gets set later
    num_update_steps_per_epoch: Optional[int] = None  # Gets set later

    global_step: int = 0
    epoch: int = 0

    best_ssim: float = 0
    best_saved_ssim: float = 0

    compile_models: List[str] = field(default_factory=list)

    # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    image_width: int = 1920
    image_height: int = 1080

    center_crop: bool = False
    flip_prob: float = 0.5

    crop_height: float = 512
    crop_width: float = 512
    resize_factor: float = 1.0

    trainable_models: List[str] = field(default_factory=list)

    loggers: List[str] = field(default_factory=list)
    wandb_project: str = "ImagineDriving"
    wandb_entity: str = "arturruiqi"
    wandb_group: str = "finetune-lora"

    output_dir: Optional[str] = None
    logging_dir: Optional[str] = None

    push_to_hub: bool = False  # Not Implemented
    hub_token: Optional[str] = None

    use_debug_metrics: bool = False
    use_recreation_loss: bool = False
    use_noise_augment: bool = True

    seed: Optional[int] = None


def init_job_id(train_state: TrainState) -> None:
    if train_state.job_id is not None:
        return

    if "SLURM_JOB_ID" in os.environ:
        train_state.job_id = os.environ["SLURM_JOB_ID"]
        return
    logging.warning(
        f"Could not find SLURM_JOB_ID or predefined job_id, setting job_id to 0"
    )
    train_state.job_id = "0"
    return


def find_checkpoint_paths(cp_dir: str, cp_prefix: str = "checkpoint", cp_delim="-"):
    cp_paths = os.listdir(cp_dir)
    cp_paths = [d for d in cp_paths if d.startswith(cp_prefix)]
    cp_paths = sorted(cp_paths, key=lambda x: int(x.split(cp_delim)[1]))
    return cp_paths


def parse_args():
    parser = ArgumentParser(
        "train_model", description="Finetune a given model on a dataset"
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--noise_strength", type=float, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--control_lora_rank", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--use_debug_metrics", action="store_true")
    parser.add_argument("--use_recreation_loss", action="store_true")
    parser.add_argument("--conditioning", nargs="*", default=None)

    args = parser.parse_args()
    return args


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def preprocess_rgb(
    batch, preprocessors, target_size: Tuple[int, int], center_crop: bool
):
    # preprocess rgb:
    # out: rgb, original_size, crop_top_left, target_size

    rgb = batch["rgb"]

    th, tw = target_size
    h0, w0 = rgb.shape[-2:]

    if resizer := preprocessors.get("resizer"):
        rgb = resizer(rgb)

    if flipper := preprocessors.get("flipper"):
        rgb = flipper(rgb)

    if cropper := preprocessors.get("cropper"):
        if center_crop:
            cy = (h0 - th) // 2
            cx = (w0 - tw) // 2
            rgb = cropper(rgb)

        else:
            cy, cx, h, w = cropper.get_params(rgb, target_size)
            rgb = transforms.functional.crop(rgb, cy, cx, h, w)
    else:
        cx, cy = 0, 0
        h, w = h0, w0

    original_size = torch.tensor((h0, w0))
    target_size = torch.tensor(target_size)
    crop_top_left = torch.tensor((cy, cx), device=rgb.device)

    return {
        "rgb": rgb,
        "original_size": original_size,
        "crop_top_left": crop_top_left,
        "target_size": target_size,
    }


def preprocess_prompt(batch, models):
    prompt = batch["prompt"]["positive_prompt"]

    assert "tokenizer" in models

    if isinstance(prompt, list):
        prompt = tuple(prompt)

    input_ids = tokenize_prompt(models["tokenizer"], prompt)[0]
    return {"input_ids": input_ids}


def preprocess_ray(
    batch: Dict[str, Any],
    ray_generators: Dict[str, RayGenerator],
    cam_to_idx: Dict[str, Dict[str, int]],
) -> Dict[str, Tensor]:
    crop_size: Tensor = batch["target_size"]
    crop_top_left: Tensor = batch["crop_top_left"]
    meta: SampleInfo = batch["meta"]

    ray_generator = ray_generators[meta.scene]
    cam_idxs = torch.tensor(
        [cam_to_idx[meta.scene][meta.sample]], device=crop_top_left.device
    )
    ray_idxs = crop_to_ray_idxs(cam_idxs, crop_top_left, crop_size)
    rays = ray_generator.forward(ray_idxs)

    ray = torch.concat([rays.origins, rays.directions], dim=-1)
    ray = ray.reshape(int(crop_size[0].item()), int(crop_size[1].item()), 6).permute(
        2, 0, 1
    )
    return {"ray": ray}


def preprocess_sample(
    batch: Dict[str, Any],
    preprocessors: Dict[str, Callable],
    preprocessor_order: List[str],
) -> Dict[str, Any]:
    sample = {}
    sample["meta"] = SampleInfo(**batch["meta"])

    for preprocess_name in preprocessor_order:

        if preprocess_name == "rgb":
            processor = preprocessors[preprocess_name]
            pp_out = processor({"rgb": batch["rgb"]})

            sample["rgb"] = pp_out["rgb"]
            sample["original_size"] = pp_out["original_size"]
            sample["crop_top_left"] = pp_out["crop_top_left"]
            sample["target_size"] = pp_out["target_size"]

        elif preprocess_name == "ref-rgb":
            if "ref-rgb" not in batch:
                continue

            ref = batch["ref-rgb"]
            sample["ref_rgb"] = ref["rgb"]
            sample["ref_gt"] = ref["gt"]

        elif preprocess_name == "prompt":
            processor = preprocessors[preprocess_name]
            pp_out = processor({"prompt": batch["prompt"]})
            sample["input_ids"] = pp_out["input_ids"]

        elif signal := ConditioningSignalInfo.from_signal_name(preprocess_name):
            processor = preprocessors[preprocess_name]
            if signal.cn_type == "rgb":
                pp_out = processor({"rgb": batch["rgb"]})
                sample[preprocess_name] = pp_out["rgb"]
                sample[preprocess_name + "_crop_top_left"] = pp_out["crop_top_left"]

            elif signal.cn_type == "ray":
                if (same_cam_rgb_name := f"cn_rgb_3_{signal.camera}") in preprocessors:
                    crop_top_left_name = same_cam_rgb_name + "_crop_top_left"
                else:
                    crop_top_left_name = "crop_top_left"

                pp_out = processor(
                    {
                        "rgb": batch["rgb"],
                        "target_size": sample["target_size"],
                        "crop_top_left": sample[crop_top_left_name],
                        "meta": sample["meta"],
                    }
                )
                sample[preprocess_name] = pp_out["ray"]

        else:
            raise NotImplementedError

    return sample


def collate_fn(
    batch: List[Dict[str, Any]], accelerator: Accelerator
) -> Dict[List[str], Any]:
    collated: dict[str, Any] = {}
    for key, item in batch[0].items():
        collated[key] = [sample[key] for sample in batch]

        if isinstance(item, (torch.Tensor, np.ndarray, int, float, bool)):
            collated[key] = torch.stack(collated[key])

        elif isinstance(item, (tuple, list)) and isinstance(item[0], (int, float)):
            collated[key] = torch.tensor(collated[key])

    for sample_name, sample in collated.items():
        if sample_name == "rgb" or sample_name.startswith("cn_rgb"):
            collated[sample_name] = sample.to(
                memory_format=torch.contiguous_format, dtype=torch.float32
            )

    return collated


def save_model_hook(
    loaded_models,
    weights,
    output_dir: str,
    accelerator: Accelerator,
    models,
    train_state: TrainState,
):
    if not accelerator.is_main_process:
        return

    layers_to_save = {}
    other_models = {}

    for loaded_model in loaded_models:
        # Map the list of loaded_models given by accelerator to keys given in train_state.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.
        # TODO: Find a better way of mapping this.

        unwrapped_model = unwrap_model(accelerator, loaded_model)
        if isinstance(
            unwrapped_model, ControlLoRAModel
        ):  # This one has special saving logic that we handle later
            other_models["controlnet"] = unwrapped_model
            continue

        state_model_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(loaded_model)
        )
        for model_name, model in models.items():
            if isinstance(unwrapped_model, type(unwrap_model(accelerator, model))):
                layers_to_save[model_name] = state_model_dict
                break

        # make sure to pop weight so that corresponding model is not saved again
        if weights:
            weights.pop()

    # TODO: Extend for more models
    dst_dir = Path(output_dir, "weights")
    if not dst_dir.exists():
        dst_dir.mkdir(exist_ok=True, parents=True)

    StableDiffusionImg2ImgPipeline.save_lora_weights(
        save_directory=str(dst_dir),
        unet_lora_layers=layers_to_save.get("unet"),
        text_encoder_lora_layers=layers_to_save.get("text_encoder"),
    )

    # TODO: Save this properly once migrated to PEFT
    if "controlnet" in other_models:
        other_models["controlnet"].save_pretrained(str(dst_dir / "controlnet"))

    configs = DiffusionModelConfig(
        train_state.model_type,
        train_state.model_id,
        lora_weights=str(dst_dir / "pytorch_lora_weights.safetensors"),
        controlnet_weights=str(dst_dir / "controlnet"),
        noise_strength=train_state.val_noise_strength,
        num_inference_steps=train_state.val_noise_num_steps,
        conditioning_signals=tuple(
            signal.name for signal in train_state.conditioning_signal_infos
        ),
    )
    save_yaml(dst_dir / "config.yml", asdict(configs))


def load_model_hook(
    loaded_models,
    input_dir,
    accelerator: Accelerator,
    models,
    train_state: TrainState,
):
    loaded_models_dict = {}

    while loaded_models:
        # Map the list of loaded_models given by accelerator to keys given in train_state.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.
        # TODO: Find a better way of mapping this.

        loaded_model = loaded_models.pop()
        for model_name, model in models.items():
            if isinstance(loaded_model, type(unwrap_model(accelerator, model))):
                loaded_models_dict[model_name] = loaded_model
                break

        else:
            raise ValueError(f"unexpected save model: {loaded_model.__class__}")

    lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(
        str(Path(input_dir) / "weights" / "pytorch_lora_weights.safetensors")
    )

    if "unet" in loaded_models_dict:
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            loaded_models_dict["unet"], unet_state_dict, adapter_name="default"
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    if "text_encoder" in loaded_models_dict:
        _set_state_dict_into_text_encoder(
            lora_state_dict,
            prefix="text_encoder.",
            text_encoder=loaded_models_dict["text_encoder"],
        )

    if "controlnet" in loaded_models_dict:
        loaded_controlnet = ControlLoRAModel.from_pretrained(
            str(Path(input_dir) / "weights"), subfolder="controlnet"
        )
        loaded_models_dict["controlnet"].load_state_dict(loaded_controlnet.state_dict())
        loaded_models_dict["controlnet"].tie_weights(models["unet"])

    # Make sure the trainable params are in float32.
    if train_state.weights_dtype in LOWER_DTYPES:
        models_to_cast = [
            loaded_model
            for model_name, loaded_model in loaded_models_dict.items()
            if model_name in train_state.trainable_models
        ]
        cast_training_params(models_to_cast, dtype=torch.float32)


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def prepare_rgb_preprocess_steps(
    train_state: TrainState, is_split_train: bool, crop_size, downsample_size
):
    steps = {}

    steps["resizer"] = transforms.Resize(
        downsample_size,
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )

    if is_split_train:
        steps["flipper"] = transforms.RandomHorizontalFlip(p=train_state.flip_prob)

        steps["cropper"] = (
            transforms.CenterCrop(crop_size)
            if train_state.center_crop
            else transforms.RandomCrop(crop_size)
        )

    else:
        steps["cropper"] = transforms.CenterCrop(crop_size)

    return steps


def prepare_preprocessors(models, train_state: TrainState):
    crop_height = train_state.crop_height or train_state.image_height
    crop_width = train_state.crop_width or train_state.image_width
    resize_factor = train_state.resize_factor or 1

    crop_size = (
        nearest_multiple(crop_height * resize_factor, 8),
        nearest_multiple(crop_width * resize_factor, 8),
    )
    downsample_size = (
        nearest_multiple(train_state.image_height * resize_factor, 8),
        nearest_multiple(train_state.image_width * resize_factor, 8),
    )

    dataset = train_state.datasets["train_data"]  # Assume train and val the same

    dataset_path = Path(dataset["path"])
    data_tree = read_data_tree(dataset["data_tree"])
    info_getter = INFO_GETTER_BUILDERS[dataset["dataset"]]()
    infos = info_getter.parse_tree(dataset_path, data_tree)
    unique_scenes = {info.scene for info in infos}
    cam_to_idx = {}
    for info in infos:
        if info.scene not in cam_to_idx:
            cam_to_idx[info.scene] = {}

        cam_to_idx[info.scene][info.sample] = len(cam_to_idx[info.scene])

    preprocessors = {"train": {}, "val": {}}

    _train_rgb_preprocess_steps = prepare_rgb_preprocess_steps(
        train_state, True, crop_size, downsample_size
    )
    _val_rgb_preprocess_steps = prepare_rgb_preprocess_steps(
        train_state, False, crop_size, downsample_size
    )

    rgb_keys = ["rgb"]
    ref_keys = ["ref-rgb"]
    ray_keys = []
    prompt_keys = ["prompt"]

    ray_cams = {}

    for signal_info in train_state.conditioning_signal_infos:
        match signal_info.cn_type:
            case "rgb":
                rgb_keys.append(signal_info.name)
            case "ray":
                ray_keys.append(signal_info.name)
                ray_cams[signal_info.name] = signal_info.camera
            case "prompt":
                prompt_keys.append(signal_info.name)
            case _:
                raise NotImplementedError

    # It is important that the Ray preprocesors occur after the RGB ones, since they need the output
    preprocessor_order = []

    for key in rgb_keys:
        preprocessor_order.append(key)
        preprocessors["train"][key] = functools.partial(
            preprocess_rgb,
            preprocessors=_train_rgb_preprocess_steps,
            target_size=crop_size,
            center_crop=train_state.center_crop,
        )

        preprocessors["val"][key] = functools.partial(
            preprocess_rgb,
            preprocessors=_val_rgb_preprocess_steps,
            target_size=crop_size,
            center_crop=True,
        )

    for key in ref_keys:
        preprocessor_order.append(key)

    for key in ray_keys:
        preprocessor_order.append(key)

        cam = ray_cams[key]
        cam_getter = CameraDataGetter(info_getter, {"camera": cam})
        cam_getter.load_cameras(
            dataset_path,
            infos,
            PoseDataGetter(info_getter, {"data_type": "pose", "camera": cam}),
            TimestampDataGetter(info_getter, {"data_type": "timestamp", "camera": cam}),
            IntrinsicsDataGetter(
                info_getter, {"data_type": "intrinsics", "camera": cam}
            ),
            img_width=train_state.image_width,
            img_height=train_state.image_height,
        )
        ray_generators = {
            scene: RayGenerator(cam_getter.cameras[scene]) for scene in unique_scenes
        }

        preprocessors["train"][key] = functools.partial(
            preprocess_ray, ray_generators=ray_generators, cam_to_idx=cam_to_idx
        )
        preprocessors["val"][key] = functools.partial(
            preprocess_ray, ray_generators=ray_generators, cam_to_idx=cam_to_idx
        )

    for key in prompt_keys:
        preprocessor_order.append(key)
        preprocessors["train"][key] = functools.partial(
            preprocess_prompt, models=models
        )

        preprocessors["val"][key] = functools.partial(preprocess_prompt, models=models)

    return preprocessors, preprocessor_order


def save_lora_weights(
    accelerator: Accelerator, models, train_state: TrainState, dir_name: str = "weights"
) -> None:
    if not accelerator.is_main_process:
        return

    trainable_models = {model: models[model] for model in train_state.trainable_models}
    layers_to_save: Dict[str, Dict[str, Union[nn.Module, Tensor]]] = {}
    other_models = {}

    for model_name, loaded_model in trainable_models.items():
        # Map the list of loaded_models given by accelerator to keys given in train_state.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.
        # TODO: Find a better way of mapping this.

        unwrapped_model = unwrap_model(accelerator, loaded_model)

        if isinstance(
            unwrapped_model, ControlLoRAModel
        ):  # This one has special saving logic that we handle later
            other_models["controlnet"] = unwrapped_model
            continue

        state_model_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(loaded_model)
        )
        layers_to_save[model_name] = state_model_dict

    dst_dir = Path(train_state.output_dir, train_state.job_id, dir_name)
    dst_dir.mkdir(exist_ok=True, parents=True)

    StableDiffusionImg2ImgPipeline.save_lora_weights(
        save_directory=str(dst_dir),
        unet_lora_layers=layers_to_save.get("unet"),
        text_encoder_lora_layers=layers_to_save.get("text_encoder"),
    )
    if "controlnet" in other_models:
        other_models["controlnet"].save_pretrained(str(dst_dir / "controlnet"))


def get_diffusion_noise(
    size: Sequence[int], device: torch.device, train_state: TrainState
) -> Tensor:
    noise = torch.randn(size, device=device)

    if train_state.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += train_state.noise_offset * torch.randn(
            (size[0], size[1], 1, 1),
            device=device,
        )
    return noise


def get_diffusion_loss(
    models, train_state: TrainState, model_input, model_pred, noise, timesteps
):
    noise_scheduler = models["noise_scheduler"]

    # Get the target for loss depending on the prediction type
    if train_state.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=train_state.prediction_type)

    match noise_scheduler.config.prediction_type:
        case "epsilon":
            target = noise
        case "v_prediction":
            # TODO: Update this with timesteps
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        case _:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

    if train_state.snr_gamma is None:
        loss = nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, train_state.snr_gamma * torch.ones_like(timesteps)],
            dim=1,
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


def save_checkpoint(accelerator: Accelerator, train_state: TrainState) -> None:
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    output_dir = os.path.join(train_state.output_dir, train_state.job_id)

    if train_state.checkpoints_total_limit is not None:
        cp_paths = find_checkpoint_paths(output_dir)

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(cp_paths) >= train_state.checkpoints_total_limit:
            num_to_remove = len(cp_paths) - train_state.checkpoints_total_limit + 1
            cps_to_remove = cp_paths[0:num_to_remove]

            logger.info(
                f"{len(cp_paths)} checkpoints already exist, removing {len(cps_to_remove)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(cps_to_remove)}")

            for cp_path in cps_to_remove:
                cp_path = os.path.join(output_dir, cp_path)
                shutil.rmtree(cp_path)

    cp_path = os.path.join(output_dir, f"checkpoint-{train_state.global_step}")
    accelerator.save_state(cp_path)
    logger.info(f"Saved state to {cp_path}")


def resume_from_checkpoint(accelerator, train_state: TrainState):
    if train_state.resume_from_checkpoint:

        if train_state.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            cp_paths = find_checkpoint_paths(
                Path(train_state.output_dir, train_state.job_id)
            )

            if len(cp_paths) == 0:
                accelerator.print(
                    f"Checkpoint '{train_state.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                return

            path = cp_paths[-1]

        else:
            path = os.path.basename(train_state.resume_from_checkpoint)

        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(Path(train_state.output_dir, train_state.job_id, path))

        train_state.global_step = int(path.split("-")[-1])
        train_state.epoch = (
            train_state.global_step // train_state.num_update_steps_per_epoch
        )

    else:
        train_state.global_step = 0


def prepare_models(train_state: TrainState, device):
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    weights_dtype = DTYPE_CONVERSION[train_state.weights_dtype]
    vae_dtype = DTYPE_CONVERSION[train_state.vae_dtype]

    models = {}
    models["noise_scheduler"] = DDPMScheduler.from_pretrained(
        train_state.model_id, subfolder="scheduler"
    )
    models["tokenizer"] = AutoTokenizer.from_pretrained(
        train_state.model_id,
        subfolder="tokenizer",
        revision=train_state.revision,
        use_fast=False,
    )
    text_encoder_cls = import_encoder_class_from_model_name_or_path(
        train_state.model_id, train_state.revision, subfolder="text_encoder"
    )
    models["text_encoder"] = text_encoder_cls.from_pretrained(
        train_state.model_id,
        subfolder="text_encoder",
        revision=train_state.revision,
        variant=train_state.variant,
    )
    models["vae"] = (
        AutoencoderKL.from_pretrained(train_state.vae_id, subfolder=None)
        if (
            train_state.vae_id
            # Need VAE fix for stable diffusion XL, see https://github.com/huggingface/diffusers/pull/4038
        )
        else AutoencoderKL.from_pretrained(
            train_state.model_id,
            subfolder="vae",
            revision=train_state.revision,
            variant=train_state.variant,
        )
    )
    models["unet"] = UNet2DConditionModel.from_pretrained(
        train_state.model_id,
        subfolder="unet",
        revision=train_state.revision,
        variant=train_state.variant,
    )

    models["image_processor"] = VaeImageProcessor()

    if train_state.model_type == "cn":
        models["controlnet"] = ControlLoRAModel.from_unet(
            unet=models["unet"],
            conditioning_channels=sum(
                info.num_channels for info in train_state.conditioning_signal_infos
            ),
            lora_linear_rank=train_state.control_lora_rank_linear,
            lora_conv2d_rank=train_state.control_lora_rank_conv2d,
            use_dora=train_state.use_dora,
        )

    # ===================
    # === Config Lora ===
    # ===================

    for model in train_state.compile_models:
        models[model] = torch.compile(models[model], backend=train_state.torch_backend)

    # We only train the additional adapter LoRA layers
    if "vae" in models:
        models["vae"].requires_grad_(False)
        models["vae"].to(device=device, dtype=vae_dtype)

    if "unet" in models:
        models["unet"].requires_grad_(False)
        models["unet"].to(device=device, dtype=weights_dtype)

        if "unet" in train_state.trainable_models:
            unet_target_ranks = train_state.lora_target_ranks["unet"]
            unet_ranks = parse_target_ranks(unet_target_ranks)

            models["unet"].add_adapter(
                LoraConfig(
                    r=train_state.lora_rank_linear,
                    lora_alpha=train_state.lora_rank_linear,
                    init_lora_weights="gaussian",
                    target_modules="|".join(unet_ranks.keys()),
                    rank_pattern=unet_ranks,
                    use_dora=train_state.use_dora,
                )
            )

    if "text_encoder" in models:
        models["text_encoder"].requires_grad_(False)
        models["text_encoder"].to(device=device, dtype=weights_dtype)

        if "text_encoder" in train_state.trainable_models:
            raise NotImplementedError
            # TODO: Update adapters
            models["text_encoder"].add_adapter(
                LoraConfig(
                    r=train_state.lora_rank,
                    lora_alpha=train_state.lora_rank,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
            )

    if "controlnet" in models:
        if "controlnet" in train_state.trainable_models:
            models["controlnet"].train()

        if "vae" in models:
            models["controlnet"].bind_vae(models["vae"])
        else:
            logger.warning(
                f"Could not find a VAE in models to bind with ControlNet. Current models: {models.keys()}"
            )

        models["controlnet"].to(device=device, dtype=weights_dtype)

    # Sanity check
    for model_name in train_state.trainable_models:
        if not model_name in models:
            raise ValueError(f"Found trainable model {model_name} not in models")

    if train_state.enable_gradient_checkpointing:
        for model_name in train_state.trainable_models:
            models[model_name].enable_gradient_checkpointing()

    # Make sure the trainable params are in float32.
    cast_training_params(
        [models[model_name] for model_name in train_state.trainable_models],
        dtype=torch.float32,
    )

    return models


@torch.no_grad()
def validate_model(
    accelerator,
    train_state: TrainState,
    dataloader: torch.utils.data.DataLoader,
    vis_dataloader: torch.utils.data.DataLoader,
    ref_dataloader: torch.utils.data.DataLoader,
    models,
    metrics: Dict[str, Any],
    run_prefix: str,
) -> None:
    assert "noise_scheduler" in models and "unet" in models

    using_wandb = any(tracker.name == "wandb" for tracker in accelerator.trackers)
    if using_wandb:
        import wandb

    weights_dtype = DTYPE_CONVERSION[train_state.weights_dtype]
    vae_dtype = DTYPE_CONVERSION[train_state.vae_dtype]

    # No need for LoRA configs, since we will pass in predefined models anyways
    diffusion_config = DiffusionModelConfig(
        model_type=train_state.model_type,
        model_id=train_state.model_id,
        noise_strength=train_state.val_noise_strength,
        num_inference_steps=train_state.val_noise_num_steps,
        conditioning_signals=tuple(train_state.conditioning_signals),
        low_mem_mode=False,
        compile_model=False,
        lora_weights=None,
    )

    pipeline_models = {"scheduler": models["noise_scheduler"]}

    if "vae" in models:
        pipeline_models["vae"] = models["vae"].to(
            dtype=vae_dtype, device=accelerator.device
        )

    if "text_encoder" in models:
        pipeline_models["text_encoder"] = unwrap_model(
            accelerator, models["text_encoder"]
        ).to(dtype=weights_dtype, device=accelerator.device)

    if "unet" in models:
        pipeline_models["unet"] = unwrap_model(accelerator, models["unet"]).to(
            dtype=weights_dtype, device=accelerator.device
        )

    if "controlnet" in models:
        pipeline_models["controlnet"] = unwrap_model(
            accelerator, models["controlnet"]
        ).to(dtype=weights_dtype, device=accelerator.device)

    if "tokenizer" in models:
        pipeline_models["tokenizer"] = models["tokenizer"]

    pipeline = DiffusionModel.from_config(
        diffusion_config,
        device=accelerator.device,
        dtype=weights_dtype,
        pipe_models=pipeline_models,
    )

    running_metrics = {
        name: torch.zeros(1, dtype=torch.float32) for name in metrics.keys()
    }
    running_count = 0

    for step, batch in enumerate(dataloader):
        diffusion_inputs = {}
        diffusion_kwargs = {"strength": train_state.val_noise_strength}

        rgb = batch["rgb"].to(dtype=vae_dtype, device=accelerator.device)
        diffusion_inputs["rgb"] = rgb
        batch_size = len(rgb)
        random_seeds = np.arange(batch_size * step, batch_size * (step + 1))

        if "controlnet" in models:
            for conditioning in train_state.conditioning_signal_infos:
                signal_name = conditioning.name

                if signal_name in batch:
                    signal_value = batch[signal_name]

                else:
                    for name, value in batch.items():
                        if (
                            signal := ConditioningSignalInfo.from_signal_name(name)
                        ) and signal.cn_type == conditioning.cn_type:
                            signal_value = value
                    else:
                        raise ValueError(
                            f"Failed to match conditioning signal {conditioning} with data {batch.keys()}"
                        )

                diffusion_inputs[signal_name] = signal_value

        diffusion_inputs["generator"] = [
            torch.Generator(device=accelerator.device).manual_seed(int(seed))
            for seed in random_seeds
        ]

        if "input_ids" in batch:
            diffusion_inputs["prompt_embeds"] = encode_tokens(
                pipeline.pipe.text_encoder,
                batch["input_ids"].to(device=accelerator.device),
                use_cache=True,
            )

        rgb_out = pipeline.get_diffusion_output(
            diffusion_inputs,
            diffusion_kwargs,
        )["rgb"]

        # Benchmark
        running_count += len(rgb)
        for metric_name, metric in metrics.items():
            values: Tensor = metric(rgb_out, rgb).detach().cpu()
            if len(values.shape) == 0:
                values = torch.tensor([values.item()])

            running_metrics[metric_name] += torch.sum(values)

            if train_state.use_debug_metrics:
                value_dict = {str(i): float(v) for i, v in enumerate(values)}
                accelerator.log(
                    {f"{run_prefix}_{metric_name}_{str(batch['meta'])}": value_dict},
                    step=train_state.global_step,
                )

    mean_metrics = {}
    for metric_name, metric in metrics.items():
        mean_metric = (running_metrics[metric_name] / running_count).item()
        mean_metrics[f"{run_prefix}_{metric_name}"] = mean_metric

        if metric_name == "ssim" and mean_metric > train_state.best_ssim:
            train_state.best_ssim = mean_metric

    accelerator.log(
        mean_metrics,
        step=train_state.global_step,
    )

    vis_rgbs = []
    vis_rgbs_noisy = []
    vis_metas = []
    vis_outs = []

    for step, batch in enumerate(vis_dataloader):
        diffusion_inputs = {}
        diffusion_kwargs = {"strength": train_state.val_noise_strength}

        rgb = batch["rgb"].to(dtype=vae_dtype, device=accelerator.device)
        diffusion_inputs["rgb"] = rgb
        batch_size = len(rgb)
        random_seeds = np.arange(batch_size * step, batch_size * (step + 1))

        if "controlnet" in models:
            for conditioning in train_state.conditioning_signal_infos:
                signal_name = conditioning.name
                if signal_name in batch:
                    diffusion_inputs[signal_name] = batch[signal_name]
                else:
                    for other_signal_name, signal in batch.items():
                        other_signal = ConditioningSignalInfo.from_signal_name(
                            other_signal_name
                        )
                        if (
                            other_signal
                            and other_signal.cn_type == conditioning.cn_type
                        ):
                            diffusion_inputs[signal_name] = signal

        diffusion_inputs["generator"] = [
            torch.Generator(device=accelerator.device).manual_seed(int(seed))
            for seed in random_seeds
        ]

        if "input_ids" in batch:
            with torch.no_grad():
                diffusion_inputs["prompt_embeds"] = encode_tokens(
                    pipeline.pipe.text_encoder,
                    batch["input_ids"].to(device=accelerator.device),
                    use_cache=True,
                )

        rgb_out = pipeline.get_diffusion_output(
            diffusion_inputs,
            diffusion_kwargs,
        )["rgb"]

        # Renders
        val_start_timestep = int(
            (1 - train_state.val_noise_strength)
            * len(models["noise_scheduler"].timesteps)
        )

        rgb_noised = get_noised_img(
            rgb,
            timestep=val_start_timestep,
            vae=pipeline.pipe.vae,
            img_processor=pipeline.pipe.image_processor,
            noise_scheduler=models["noise_scheduler"],
        )

        vis_rgbs.extend(list(rgb.detach().cpu()))
        vis_outs.extend(list(rgb_out.detach().cpu()))
        vis_rgbs_noisy.extend(list(rgb_noised.detach().cpu()))
        vis_metas.extend(batch["meta"])

    ref_neurad_running_metrics = {
        name: torch.zeros(1, dtype=torch.float32) for name in metrics.keys()
    }
    ref_running_metrics = {
        name: torch.zeros(1, dtype=torch.float32) for name in metrics.keys()
    }
    ref_running_count = 0
    ref_gts = []
    ref_rgbs = []
    ref_outs = []
    ref_metas = []

    for step, batch in enumerate(ref_dataloader):
        diffusion_inputs = {}
        diffusion_kwargs = {"strength": train_state.val_noise_strength}

        ref_rgb = batch["ref_rgb"].to(dtype=vae_dtype, device=accelerator.device)
        ref_gt = batch["ref_gt"].to(dtype=vae_dtype, device=accelerator.device)

        diffusion_inputs["rgb"] = ref_rgb
        batch_size = len(ref_rgb)
        random_seeds = np.arange(batch_size * step, batch_size * (step + 1))

        if "controlnet" in models:
            for conditioning in train_state.conditioning_signal_infos:
                signal_name = conditioning.name
                if signal_name in batch:
                    diffusion_inputs[signal_name] = batch[signal_name]
                else:
                    for other_signal_name, signal in batch.items():
                        other_signal = ConditioningSignalInfo.from_signal_name(
                            other_signal_name
                        )
                        if (
                            other_signal
                            and other_signal.cn_type == conditioning.cn_type
                        ):
                            diffusion_inputs[signal_name] = signal

        diffusion_inputs["generator"] = [
            torch.Generator(device=accelerator.device).manual_seed(int(seed))
            for seed in random_seeds
        ]

        if "input_ids" in batch:
            with torch.no_grad():
                diffusion_inputs["prompt_embeds"] = encode_tokens(
                    pipeline.pipe.text_encoder,
                    batch["input_ids"].to(device=accelerator.device),
                    use_cache=True,
                )

        ref_out = pipeline.get_diffusion_output(
            diffusion_inputs,
            diffusion_kwargs,
        )["rgb"]

        ref_running_count += len(rgb)
        for metric_name, metric in metrics.items():
            values: Tensor = metric(ref_out, ref_gt).detach().cpu()
            if len(values.shape) == 0:
                values = torch.tensor([values.item()])

            ref_running_metrics[metric_name] += torch.sum(values)

            if train_state.use_debug_metrics:
                value_dict = {str(i): float(v) for i, v in enumerate(values)}
                accelerator.log(
                    {
                        f"{run_prefix}_ref_{metric_name}_{str(batch['meta'])}": value_dict
                    },
                    step=train_state.global_step,
                )

            values_neurad: Tensor = metric(ref_rgb, ref_gt).detach().cpu()
            if len(values_neurad.shape) == 0:
                values_neurad = torch.tensor([values_neurad.item()])

            ref_neurad_running_metrics[metric_name] += torch.sum(values_neurad)

            if train_state.use_debug_metrics:
                value_dict = {str(i): float(v) for i, v in enumerate(values_neurad)}
                accelerator.log(
                    {
                        f"{run_prefix}_ref_neurad_{metric_name}_{str(batch['meta'])}": value_dict
                    },
                    step=train_state.global_step,
                )

        ref_rgbs.extend(list(ref_rgb.detach().cpu()))
        ref_outs.extend(list(ref_out.detach().cpu()))
        ref_gts.extend(list(ref_gt.detach().cpu()))
        ref_metas.extend(batch["meta"])

    mean_ref_metrics = {}
    mean_ref_neurad_metrics = {}
    for metric_name, metric in metrics.items():
        mean_ref_metrics[f"{run_prefix}_ref_{metric_name}"] = (
            ref_running_metrics[metric_name] / ref_running_count
        ).item()
        mean_ref_neurad_metrics[f"{run_prefix}_ref_neurad_{metric_name}"] = (
            ref_neurad_running_metrics[metric_name] / ref_running_count
        ).item()

    accelerator.log(
        mean_ref_metrics,
        step=train_state.global_step,
    )
    accelerator.log(
        mean_ref_neurad_metrics,
        step=train_state.global_step,
    )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(
                f"{run_prefix}_images",
                np.stack([np.asarray(img) for img in vis_outs]),
                train_state.epoch,
                dataformats="NHWC",
            )

        if tracker.name == "wandb" and using_wandb:
            tracker.log_images(
                {
                    "ground_truth": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(vis_rgbs, vis_metas))
                    ],
                    f"{run_prefix}_images": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(vis_outs, vis_metas))
                    ],
                    "noised_images": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(vis_rgbs_noisy, vis_metas))
                    ],
                    "ref_ground_truth": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(ref_gt, ref_metas))
                    ],
                    "ref_neurad": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(ref_rgb, ref_metas))
                    ],
                    f"ref_{run_prefix}": [
                        wandb.Image(
                            img,
                            caption=str(meta),
                        )
                        for (img, meta) in (zip(ref_out, ref_metas))
                    ],
                },
                step=train_state.global_step,
            )


def train_epoch(
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: TrainState,
    dataloader: torch.utils.data.DataLoader,
    models,
    params_to_optimize,
    progress_bar: tqdm.tqdm,
):
    weights_dtype = DTYPE_CONVERSION[train_state.weights_dtype]
    vae_dtype = DTYPE_CONVERSION[train_state.vae_dtype]
    noise_scheduler = models["noise_scheduler"]

    if "unet" in train_state.trainable_models:
        models["unet"].train()

    if "text_encoder" in train_state.trainable_models:
        models["text_encoder"].train()

    train_loss = 0.0

    for i_batch, batch in enumerate(dataloader):
        if i_batch >= train_state.num_update_steps_per_epoch:
            break

        assert "rgb" in batch

        with accelerator.accumulate(models[m] for m in train_state.trainable_models):
            rgb = batch["rgb"].to(dtype=vae_dtype, device=accelerator.device)
            rgb_gt = rgb

            if train_state.use_noise_augment:
                rgb = generate_noise_pattern(
                    n_clusters=256,
                    cluster_size_min=2,
                    cluster_size_max=8,
                    noise_strength=0.2,
                    pattern=rgb,
                )

            rgb = models["image_processor"].preprocess(rgb)

            model_input = (
                models["vae"].encode(rgb).latent_dist.sample()
                * models["vae"].config.scaling_factor
            ).to(weights_dtype)

            noise = get_diffusion_noise(
                model_input.shape, model_input.device, train_state
            )

            unet_added_conditions = {}
            unet_kwargs = {}

            timesteps = get_random_timesteps(
                train_state.train_noise_strength,
                noise_scheduler.config.num_train_timesteps,
                model_input.device,
                model_input.size(0),
                low_noise_high_step=False,
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            prompt_hidden_state = encode_tokens(
                models["text_encoder"],
                batch["input_ids"],
                use_cache="text_encoder" not in train_state.trainable_models,
            )

            if "controlnet" in models:
                conditioning = combine_conditioning_info(
                    batch, train_state.conditioning_signal_infos
                ).to(noisy_model_input.device)

                down_block_res_samples, mid_block_res_sample = models["controlnet"](
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_hidden_state,
                    controlnet_cond=conditioning,
                    return_dict=False,
                )
                unet_kwargs["down_block_additional_residuals"] = [
                    sample.to(dtype=weights_dtype) for sample in down_block_res_samples
                ]
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample.to(
                    dtype=weights_dtype
                )

            model_pred = models["unet"](
                noisy_model_input,
                timesteps,
                prompt_hidden_state,
                added_cond_kwargs=unet_added_conditions,
                **unet_kwargs,
            ).sample

            loss = get_diffusion_loss(
                models, train_state, model_input, model_pred, noise, timesteps
            )

            if train_state.use_recreation_loss:
                # TODO: Completely change this
                pred_rgb = decode_img(
                    models["image_processor"], models["vae"], noisy_model_input
                )
                rec_loss = (
                    torch.mean(
                        train_state.rec_loss_strength
                        * nn.functional.mse_loss(pred_rgb, rgb_gt, reduce=None)
                        * (1 - (timesteps + 1) / noise_scheduler.num_train_timesteps)
                    )
                    ** 2
                )
                loss += rec_loss

            if train_state.use_debug_metrics and "meta" in batch:
                for meta in batch["meta"]:
                    accelerator.log(
                        {"loss_for_" + str(meta): loss},
                        step=train_state.global_step,
                    )

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(
                loss.repeat(train_state.train_batch_size)
            ).mean()
            train_loss += avg_loss.item() / train_state.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    params_to_optimize, train_state.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            train_state.global_step += 1
            accelerator.log({"train_loss": train_loss}, step=train_state.global_step)
            train_loss = 0.0

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    train_state.epoch += 1


def main(args: Namespace) -> None:
    # ========================
    # ===   Setup script   ===
    # ========================

    config = setup_project(args.config_path)

    train_state = TrainState(
        **config
    )  # TODO: Implement `from_config` to enable support for dataclasses

    if args.n_epochs is not None:
        train_state.n_epochs = args.n_epochs

    if args.noise_strength is not None:
        train_state.train_noise_strength = args.noise_strength
        train_state.val_noise_strength = args.noise_strength

    if args.scene is not None:
        new_datasets = {}
        for dataset_name, dataset in train_state.datasets:
            scene_list = list(dataset.keys())
            if not len(scene_list) == 1:
                raise ValueError(
                    f"Cannot override scene if there is more than one scene per dataset"
                )

            new_datasets[dataset_name] = {}
            for scene in scene_list:
                new_datasets[dataset_name][args.scene] = dataset[scene]
        train_state.datasets = new_datasets

    if args.model_type is not None:
        train_state.model_type = args.model_type
        if args.model_type == "cn" and "controlnet" not in train_state.trainable_models:
            train_state.trainable_models.append("controlnet")
        elif args.model_type == "sd" and "controlnet" in train_state.trainable_models:
            train_state.trainable_models.pop(
                train_state.trainable_models.index("controlnet")
            )

        if args.model_type == "sd":
            train_state.conditioning_signals = []
            train_state.conditioning_signal_infos = []

        if (
            args.model_type == "cn"
            and len(train_state.conditioning_signals) == 0
            and (args.conditioning is None)
        ):
            raise ValueError(f"Cannot run controlnet with no conditioning signals")

    if args.snr_gamma is not None:
        train_state.snr_gamma = args.snr_gamma

    if args.learning_rate is not None:
        train_state.learning_rate = args.learning_rate

    if args.lora_rank is not None:
        train_state.lora_rank_linear = args.lora_rank
        train_state.lora_rank_conv2d = args.lora_rank
        for model_name, model in train_state.lora_target_ranks.items():
            for block_name, block in model.items():
                for layer_name, layer_rank in block.items():
                    train_state.lora_target_ranks[model_name][block_name][
                        layer_name
                    ] = args.lora_rank

    if args.control_lora_rank is not None:
        train_state.control_lora_rank_linear = args.control_lora_rank
        train_state.control_lora_rank_conv2d = args.control_lora_rank

    if args.dataloader_num_workers is not None:
        train_state.dataloader_num_workers = args.dataloader_num_workers

    if args.use_debug_metrics:
        train_state.use_debug_metrics = True

    if args.use_recreation_loss:
        train_state.use_recreation_loss = True

    if args.conditioning is not None:
        train_state.conditioning_signals = args.conditioning

    accelerator = Accelerator(
        gradient_accumulation_steps=train_state.gradient_accumulation_steps,
        mixed_precision=(
            train_state.weights_dtype
            if train_state.weights_dtype in LOWER_DTYPES
            else "no"
        ),
        log_with=train_state.loggers,
        project_config=ProjectConfiguration(
            project_dir=train_state.output_dir,
            logging_dir=train_state.logging_dir,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    init_job_id(train_state)
    logger.info(f"Launching script under id: {train_state.job_id}")
    logger.info(
        f"Number of cuda detected devices: {torch.cuda.device_count()}, Using device: {accelerator.device}, distributed: {accelerator.distributed_type}"
    )

    train_state.conditioning_signal_infos = [
        ConditioningSignalInfo.from_signal_name(signal)
        for signal in train_state.conditioning_signals
    ]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if train_state.seed is not None:
        set_seed(train_state.seed)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # Note: this applies to the A100
    if train_state.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if torch.backends.mps.is_available() and train_state.weights_dtype == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if accelerator.is_main_process:
        Path(train_state.output_dir).mkdir(exist_ok=True, parents=True)
        Path(train_state.output_dir, train_state.job_id).mkdir(exist_ok=True)
        Path(train_state.logging_dir).mkdir(exist_ok=True)

        if train_state.push_to_hub:
            raise NotImplementedError

    if "wandb" in train_state.loggers:
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

        if train_state.hub_token is not None:
            raise ValueError(
                "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        import wandb

        if accelerator.is_local_main_process:
            wandb.init(
                project=train_state.wandb_project or get_env("WANDB_PROJECT"),
                entity=train_state.wandb_entity or get_env("WANDB_ENTITY"),
                dir=(
                    train_state.logging_dir
                    if train_state.logging_dir
                    else get_env("WANDB_DIR")
                ),
                group=train_state.wandb_group or get_env("WANDB_GROUP"),
                reinit=True,
                config=asdict(train_state),
            )

    # =======================
    # ===   Load models   ===
    # =======================

    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(
            data_range=(0.0, 1.0), reduction="none"
        ).to(accelerator.device),
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device),
        "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            accelerator.device
        ),
    }

    models = prepare_models(train_state, accelerator.device)

    # ============================
    # === Prepare optimization ===
    # ============================

    accelerator.register_save_state_pre_hook(
        functools.partial(
            save_model_hook,
            accelerator=accelerator,
            models=models,
            train_state=train_state,
        )
    )
    accelerator.register_load_state_pre_hook(
        functools.partial(
            load_model_hook,
            accelerator=accelerator,
            models=models,
            train_state=train_state,
        )
    )

    # ======================
    # ===   Setup data   ===
    # ======================

    preprocessors, key_order = prepare_preprocessors(models, train_state)
    train_dataset = DynamicDataset.from_config(
        train_state.datasets["train_data"],
        preprocess_func=functools.partial(
            preprocess_sample,
            preprocessors=preprocessors["train"],
            preprocessor_order=key_order,
        ),
    )
    val_dataset = DynamicDataset.from_config(
        train_state.datasets["val_data"],
        preprocess_func=functools.partial(
            preprocess_sample,
            preprocessors=preprocessors["val"],
            preprocessor_order=key_order,
        ),
    )

    render_dataset = DynamicDataset.from_config(
        train_state.datasets["render_data"],
        preprocess_func=functools.partial(
            preprocess_sample,
            preprocessors=preprocessors["val"],
            preprocessor_order=key_order,
        ),
    )

    ref_dataset = DynamicDataset.from_config(
        train_state.datasets["ref_data"],
        preprocess_func=functools.partial(
            preprocess_sample,
            preprocessors=preprocessors["val"],
            preprocessor_order=key_order,
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=functools.partial(collate_fn, accelerator=accelerator),
        pin_memory=train_state.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=functools.partial(collate_fn, accelerator=accelerator),
        pin_memory=train_state.pin_memory,
    )

    render_dataloader = DataLoader(
        render_dataset,
        shuffle=False,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=functools.partial(collate_fn, accelerator=accelerator),
        pin_memory=train_state.pin_memory,
    )

    ref_dataloader = DataLoader(
        ref_dataset,
        shuffle=False,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=functools.partial(collate_fn, accelerator=accelerator),
        pin_memory=train_state.pin_memory,
    )

    # ==========================
    # ===   Setup training   ===
    # ==========================

    trainable_models = [
        models[model_name] for model_name in train_state.trainable_models
    ]
    params_to_optimize = list(
        filter(
            lambda p: p.requires_grad,
            it.chain(*(model.parameters() for model in trainable_models)),
        )
    )

    if train_state.scale_lr:
        train_state.learning_rate = (
            train_state.learning_rate
            * train_state.gradient_accumulation_steps
            * train_state.train_batch_size
            * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=train_state.learning_rate,
        betas=(train_state.adam_beta1, train_state.adam_beta2),
        weight_decay=train_state.adam_weight_decay,
        eps=train_state.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    train_state.num_update_steps_per_epoch = math.ceil(
        train_state.frac_dataset_per_epoch
        * len(train_dataloader)
        / train_state.gradient_accumulation_steps
    )
    train_state.max_train_steps = (
        train_state.n_epochs * train_state.num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        train_state.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_state.lr_warmup_steps
        * train_state.gradient_accumulation_steps,
        num_training_steps=train_state.max_train_steps
        * train_state.gradient_accumulation_steps,
        **train_state.lr_scheduler_kwargs,
    )

    optimizer, train_dataloader, lr_scheduler, *trainable_models = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, *trainable_models
    )
    for model, model_name in zip(trainable_models, train_state.trainable_models):
        models[model_name] = model

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    train_state.num_update_steps_per_epoch = math.ceil(
        train_state.frac_dataset_per_epoch
        * len(train_dataloader)
        / train_state.gradient_accumulation_steps
    )
    train_state.max_train_steps = (
        train_state.n_epochs * train_state.num_update_steps_per_epoch
    )

    # Afterwards we recalculate our number of training epochs
    train_state.n_epochs = math.ceil(
        train_state.max_train_steps / train_state.num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(train_state.wandb_group, config=asdict(train_state))

    # ===================
    # === Train model ===
    # ===================
    total_batch_size = (
        train_state.train_batch_size
        * accelerator.num_processes
        * train_state.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num training samples = {len(train_dataset)}")
    logger.info(f"  Num validation samples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {train_state.n_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {train_state.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_state.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {train_state.max_train_steps}")
    logger.info(f"Configuration: {train_state}")

    if train_state.resume_from_checkpoint:
        resume_from_checkpoint(accelerator, train_state)

    progress_bar = tqdm.tqdm(
        range(0, train_state.max_train_steps),
        initial=train_state.global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    while train_state.epoch < train_state.n_epochs:
        train_epoch(
            accelerator,
            optimizer,
            lr_scheduler,
            train_state,
            train_dataloader,
            models,
            params_to_optimize,
            progress_bar,
        )
        torch.cuda.empty_cache()

        if train_state.epoch % train_state.val_freq == 0:
            validate_model(
                accelerator,
                train_state,
                val_dataloader,
                render_dataloader,
                ref_dataloader,
                models,
                metrics,
                "val",
            )
        torch.cuda.empty_cache()

        # if accelerator.is_main_process and (
        #    train_state.global_step % train_state.checkpointing_steps == 0
        # ):
        if accelerator.is_main_process:
            if (
                train_state.checkpoint_strategy == "best"
                and train_state.best_ssim > train_state.best_saved_ssim
            ):
                save_checkpoint(accelerator, train_state)
                train_state.best_saved_ssim = train_state.best_ssim

            elif (
                train_state.checkpoint_strategy == "latest"
                and train_state.global_step % train_state.checkpointing_steps == 0
            ):
                save_checkpoint(accelerator, train_state)

        if (
            train_state.global_step >= train_state.max_train_steps
            or train_state.epoch > train_state.n_epochs
        ):
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Final inference
        validate_model(
            accelerator,
            train_state,
            val_dataloader,
            render_dataloader,
            ref_dataloader,
            models,
            metrics,
            "test",
        )

        # Save the lora layers
        save_lora_weights(accelerator, models, train_state, "final_weights")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
