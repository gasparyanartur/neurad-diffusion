# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Literal,
    Tuple,
    List,
    Optional,
    Type,
    Union,
    Sequence,
    cast,
)
from collections.abc import Iterable
from pathlib import Path
import logging
import os
from dataclasses import asdict
import math
import itertools as it
from typing_extensions import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
import torch.utils
import torch.utils.data
import tqdm
import shutil
import functools

import numpy as np
import torch
import tyro
import wandb
from torch import FloatTensor, IntTensor, nn
from torch import Tensor
from torch.utils.data import DataLoader
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import PeftModel

from nerfstudio.generative.diffusion_model import (
    ConditioningSignalInfo,
    Metrics,
    StableDiffusionModel,
    combine_conditioning_info,
    encode_img,
    is_metric_improved,
)
from nerfstudio.generative.dynamic_dataset import (
    DataGetter,
    DataSpec,
    DataSpecT,
    DatasetConfig,
    DatasetTree,
    InfoGetter,
    LidarDataSpec,
    NerfOutputSpec,
    PromptDataSpec,
    RayDataSpec,
    RgbDataSpec,
    SampleConfig,
    is_data_spec_type_rgb,
    iter_numeric_names,
    read_yaml,
    save_yaml,
    setup_cache,
    DynamicDataset,
)
from nerfstudio.generative.diffusion_model import (
    get_noised_img,
    encode_tokens,
    DiffusionModelConfig,
    decode_img,
    get_random_timesteps,
    DiffusionModelType,
    DiffusionModelId,
)
from nerfstudio.generative.utils import (
    LOWER_DTYPES,
    get_env,
)

check_min_version("0.27.0")
logger = get_logger(__name__, log_level="INFO")


DATASET_TYPE_T = Union[
    Literal["train"], Literal["val"], Literal["render"], Literal["nerf_out"]
]


class TrainingDatasetConfigs(BaseModel):
    type: str

    def get_dataset_and_loader(
        self,
        dataset_type: DATASET_TYPE_T,
        num_workers: int,
        batch_size: int,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
    ) -> Union[Tuple[DynamicDataset, DataLoader], Tuple[None, None]]:
        raise NotImplementedError


class FullTrainingDatasetConfig(BaseModel):
    type: Literal["full"] = "full"

    train_dataset: DatasetConfig
    val_dataset: DatasetConfig
    render_dataset: Optional[DatasetConfig] = None
    nerf_out_dataset: Optional[DatasetConfig] = None

    def make_dataset_loader(
        self,
        dataset_type: DATASET_TYPE_T,
        num_workers: int,
        batch_size: int,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
        info_getter: Optional[InfoGetter] = None,
        data_getters: Optional[Dict[str, DataGetter]] = None,
    ) -> Union[Tuple[DynamicDataset, DataLoader], Tuple[None, None]]:

        if dataset_type == "train":
            dataset_config = self.train_dataset

        elif dataset_type == "val":
            dataset_config = self.val_dataset

        elif dataset_type == "render":
            if not self.render_dataset:
                return None, None

            dataset_config = self.render_dataset

        elif dataset_type == "nerf_out":
            if not self.nerf_out_dataset:
                return None, None

            dataset_config = self.nerf_out_dataset

        else:
            raise ValueError(f"Unknown dataset type {dataset_type}")

        if shuffle is None:
            shuffle = dataset_type == "train"

        dataset = DynamicDataset(dataset_config, info_getter, data_getters)

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=functools.partial(
                collate_fn, data_specs=dataset.dataset_config.data_specs
            ),
        )

        return dataset, dataloader


class SingleSceneTrainingDatasetConfig(BaseModel):
    type: Literal["single_scene"] = "single_scene"

    camera: str = "front_camera"
    ref_camera: str = "front_left_camera"

    dataset_path: Path = Path("data", "pandaset")
    nerf_output_dir: Path = Path("data", "nerf_outputs")

    random_crop: bool = False
    conditioning: Tuple[str, ...] = ("ray",)

    dataset: str = "pandaset"
    scene: str = "001"
    sample_start: str = "00"
    sample_end: str = "80"
    sample_step: int = 1

    use_render: bool = True
    use_nerf_out: bool = True

    def make_dataset_loader(
        self,
        dataset_type: DATASET_TYPE_T,
        num_workers: int,
        batch_size: int,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
        info_getter: Optional[InfoGetter] = None,
        data_getters: Optional[Dict[str, DataGetter]] = None,
    ) -> Union[Tuple[DynamicDataset, DataLoader], Tuple[None, None]]:

        shuffle = shuffle if (shuffle is not None) else (dataset_type == "train")
        split = "train" if dataset_type == "train" else "test"
        camera = self.camera if dataset_type != "nerf_out" else self.ref_camera

        if (dataset_type == "render" and not self.use_render) or (
            dataset_type == "nerf_out" and not self.use_nerf_out
        ):
            return None, None

        data_specs = {
            "rgb": RgbDataSpec(camera=camera),
            "input_ids": PromptDataSpec(),
        }

        if dataset_type == "nerf_out":
            data_specs["rgb_gt"] = RgbDataSpec(camera=camera)
            data_specs["rgb_nerf"] = NerfOutputSpec(
                camera=camera, nerf_output_path=str(self.nerf_output_dir)
            )
        else:
            data_specs["rgb"] = RgbDataSpec(camera=camera)

        if "ray" in self.conditioning:
            data_specs["ray"] = RayDataSpec(camera=camera)

        samples = list(
            iter_numeric_names(self.sample_start, self.sample_end, self.sample_step)
        )

        dataset_config = DatasetConfig(
            dataset_path=self.dataset_path,
            data_specs=cast(
                Dict[str, DataSpecT],
                data_specs,
            ),
            sample_config=SampleConfig(random_crop=self.random_crop),
            data_tree=DatasetTree.single_split_single_scene(
                self.dataset, split, self.scene, samples
            ),
        )

        dataset = DynamicDataset(dataset_config, info_getter, data_getters)

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=functools.partial(
                collate_fn, data_specs=dataset.dataset_config.data_specs
            ),
        )

        return dataset, dataloader


TrainingDatasetConfigT = Annotated[
    Union[FullTrainingDatasetConfig, SingleSceneTrainingDatasetConfig],
    Field(discriminator="type"),
]


def make_dataset_loader(
    dataset: DynamicDataset,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DynamicDataset, DataLoader]:

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=functools.partial(
            collate_fn, data_specs=dataset.dataset_config.data_specs
        ),
    )

    return dataset, dataloader


class TrainConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="lora_train_",
        cli_parse_args=True,
        yaml_file=os.getenv("LORA_TRAIN_CONFIG", None),
    )

    output_dir: Path = Field(default=Path("outputs", "train-lora"))
    cache_dir: Path = Field(default=Path(".cache", "train-lora"))
    logging_dir: Path = Field(default=Path("logs", "train-lora"))

    job_id: str = Field(default="0", alias="slurm_job_id")
    project_name: str = "diffusion-nerf"

    checkpoint_path: Optional[str] = None

    noise_scheduler_prediction_type: Optional[str] = None
    enable_gradient_checkpointing: bool = False

    checkpoint_strategy: str = "best"
    checkpointing_steps: int = 500
    max_num_checkpoints: int = (
        0  # How many checkpoints, besides the latest, that will be stored
    )
    checkpoint_metric: str = Metrics.lpips

    n_epochs: int = 1000
    max_train_samples: Optional[int] = None
    val_freq: int = 50
    frac_dataset_per_epoch: float = 1.0

    allow_tf32: bool = (
        True  # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    )
    torch_backend: str = "cudagraphs"

    scale_lr: bool = False
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    snr_gamma: Optional[float] = None
    max_grad_norm: float = 1.0
    noise_offset: float = (
        0  # https://www.crosslabs.org//blog/diffusion-with-offset-noise
    )

    rec_loss_strength: float = 0.1

    # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 1000
    lr_scheduler_kwargs: Dict[str, Any] = {}

    loggers: List[str] = ["wandb"]
    wandb_project: str = "diffusion-nerf"
    wandb_entity: str = "arturruiqi"
    wandb_group: str = "finetune-lora"

    push_to_hub: bool = False  # Not Implemented
    hub_token: Optional[str] = None

    use_debug_metrics: bool = False
    use_recreation_loss: bool = False
    use_noise_augment: bool = False

    seed: Optional[int] = 0

    train_noise_strength: float = 0.1
    train_noise_num_steps: Optional[int] = None
    use_cached_tokens: bool = True

    diffusion_config: DiffusionModelConfig = DiffusionModelConfig(
        type=DiffusionModelType.cn,
        id=DiffusionModelId.sd_v2_1,
        dtype="fp32",
        noise_strength=0.1,
        num_inference_steps=50,
        enable_progress_bar=False,
        lora_weights=None,
        models_to_train_lora=("unet",),
        models_to_load_lora=(),
        use_dora=True,
        lora_model_prefix="lora_",
        conditioning_signals=("ray",),
        guidance_scale=0,
        metrics=("lpips", "ssim", "psnr", "mse"),
    )

    dataset_configs: TrainingDatasetConfigT = SingleSceneTrainingDatasetConfig(
        camera="front_camera",
        ref_camera="front_left_camera",
        conditioning=("ray",),
        random_crop=False,
        dataset="pandaset",
        scene="001",
        sample_start="00",
        sample_end="00",
        sample_step=1,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class TrainState(BaseModel):
    max_train_steps: int = -1
    num_update_steps_per_epoch: int = -1

    global_step: int = 0
    epoch: int = 0

    loss_history: List[float] = []

    best_metric: float = 0
    best_saved_metric: float = 0

    @property
    def max_epoch(self) -> int:
        return self.max_train_steps // self.num_update_steps_per_epoch


def unwrap_model(accelerator: Accelerator, model, unpeft: bool = True):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model

    if unpeft and isinstance(model, PeftModel):
        model = model.base_model.model

    return model


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


def collate_fn(
    batch: List[Dict[str, Any]], data_specs: Dict[str, DataSpecT]
) -> Dict[str, Iterable[Any]]:
    assert len(batch) > 0

    collated: dict[str, Iterable[Any]] = {}

    for key in batch[0].keys():
        spec = data_specs.get(key)

        item = [sample[key] for sample in batch]

        if isinstance(item[0], (list, tuple, np.ndarray, int, float, bool)):
            item = list(map(torch.tensor, item))

        if isinstance(item[0], torch.Tensor):
            item = torch.stack(item)

            if spec and is_data_spec_type_rgb(type(spec)):
                item = item.contiguous(memory_format=torch.contiguous_format)

        collated[key] = item

    return collated


def is_model_equal_type(accelerator, model_1, model_2, unwrap: bool = True) -> bool:
    if unwrap:
        model_1 = unwrap_model(accelerator, model_1)
        model_2 = unwrap_model(accelerator, model_2)

    return isinstance(model_1, type(model_2))


def find_matching_model(
    accelerator: Accelerator, accelerator_model, diffusion_model: StableDiffusionModel
):
    # Map the list of loaded_models given by accelerator to keys given in train_config.
    # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.

    for model_name, other_model in diffusion_model.models.items():
        if is_model_equal_type(accelerator, accelerator_model, other_model):
            return model_name

    return None


def save_lora_weights(
    accelerator: Accelerator,
    diffusion_model: StableDiffusionModel,
    train_config: TrainConfig,
    dir_name: str = "weights",
) -> None:
    if not accelerator.is_main_process:
        return

    dst_dir = train_config.output_dir / train_config.job_id / dir_name
    dst_dir.mkdir(exist_ok=True, parents=True)

    models_to_save = [
        unwrap_model(accelerator, model, unpeft=False)
        for model in diffusion_model.get_models_to_train()
    ]

    assert all(isinstance(model, PeftModel) for model in models_to_save)

    for model in models_to_save:
        model.save_pretrained(str(dst_dir))


def save_model_hook(
    loaded_models,
    weights,
    output_dir: str,
    accelerator: Accelerator,
    diffusion_model: StableDiffusionModel,
    train_config: TrainConfig,
    train_state: TrainState,
):
    if not accelerator.is_main_process:
        return

    dst_dir = Path(output_dir, "weights")
    if not dst_dir.exists():
        dst_dir.mkdir(exist_ok=True, parents=True)

    for loaded_accelerator_model in loaded_models:
        # make sure to pop weight so that corresponding model is not saved again
        if weights:
            weights.pop()

        matching_model_name = find_matching_model(
            accelerator, loaded_accelerator_model, diffusion_model
        )
        if not matching_model_name:
            continue

        model = unwrap_model(accelerator, loaded_accelerator_model, unpeft=False)
        assert isinstance(model, PeftModel)

        model.save_pretrained(str(dst_dir))

    save_yaml(dst_dir / "config.yml", train_config.model_dump())
    save_yaml(dst_dir / "state.yml", train_state.model_dump())


def load_model_hook(
    loaded_models: List[PeftModel],
    input_dir: Path,
    accelerator: Accelerator,
    diffusion_model: StableDiffusionModel,
    train_config: TrainConfig,
    train_state: TrainState,
):
    loaded_models_dict = {}
    while loaded_models:
        # Map the list of loaded_models given by accelerator to keys given in train_config.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.

        loaded_model = loaded_models.pop()
        matching_model_name = find_matching_model(
            accelerator, loaded_model, diffusion_model
        )
        if not matching_model_name:
            raise ValueError(f"unexpected save model: {type(loaded_model)}")

        loaded_models_dict[matching_model_name] = loaded_model

    weights_dir = input_dir / "weights"
    for model_dir in weights_dir.iterdir():
        model_name = model_dir.name

        # Assuming adapter name is same as model name prefixed with lora_ (i.e. lora_unet, lora_text_encoder, lora_controlnet)
        if not model_name.startswith(train_config.diffusion_config.lora_model_prefix):
            continue

        model_name = model_name[len(train_config.diffusion_config.lora_model_prefix) :]

        if model_name not in loaded_models_dict:
            logger.warning(
                f"Found weights {model_name} in weight directory, but model not found in {train_config.diffusion_config.models_to_train_lora}. Skipping"
            )
        model = loaded_models_dict[model_name]
        loaded_models_dict[model_name] = PeftModel.from_pretrained(
            model, str(model_dir)
        )

    # Make sure the trainable params are in float32.
    if train_config.diffusion_config.dtype in LOWER_DTYPES:
        models_to_cast = [
            loaded_model
            for model_name, loaded_model in loaded_models_dict.items()
            if model_name in train_config.diffusion_config.models_to_train_lora
        ]
        cast_training_params(models_to_cast, dtype=torch.float32)

    state = read_yaml(input_dir / "state.yml")
    for f in train_state.model_fields.keys():
        if f in state:
            setattr(train_state, f, state[f])


def save_checkpoint(
    accelerator: Accelerator, train_config: TrainConfig, train_state: TrainState
) -> None:
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    output_dir = train_config.output_dir / train_config.job_id

    if train_config.max_num_checkpoints is not None:
        cp_paths = find_checkpoint_paths(output_dir)

        # before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
        if len(cp_paths) >= train_config.max_num_checkpoints:
            # Remove one more to make space for the new one
            num_to_remove = len(cp_paths) - train_config.max_num_checkpoints + 1
            cps_to_remove = cp_paths[:num_to_remove]

            logger.info(
                f"{len(cp_paths)} checkpoints already exist, removing {len(cps_to_remove)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(map(str, cps_to_remove))}")

            for cp_path in cps_to_remove:
                shutil.rmtree(cp_path)

    cp_path = output_dir / f"checkpoint-{train_state.global_step}"
    accelerator.save_state(str(cp_path))
    logger.info(f"Saved state to path {cp_path}")


def resume_from_checkpoint(
    accelerator, train_config: TrainConfig, train_state: TrainState
):
    if train_config.checkpoint_path is None:
        train_state.global_step = 0
        return

    if train_config.checkpoint_path == "latest":
        # Get the most recent checkpoint
        cp_paths = find_checkpoint_paths(train_config.output_dir / train_config.job_id)

        if len(cp_paths) == 0:
            accelerator.print(
                f"Checkpoint '{train_config.checkpoint_path}' does not exist. Starting a new training run."
            )
            return

        path = cp_paths[-1]

    else:
        path = Path(train_config.checkpoint_path)

    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(path)

    train_state.global_step = int(path.name.split("-")[-1])
    train_state.epoch = (
        train_state.global_step // train_state.num_update_steps_per_epoch
    )


def find_checkpoint_paths(
    path: Path, cp_prefix: str = "checkpoint", cp_delim="-"
) -> List[Path]:
    cp_paths = [d for d in path.iterdir() if d.name.startswith(cp_prefix)]
    cp_paths = sorted(cp_paths, key=lambda d: int(d.name.split(cp_delim)[1]))
    return cp_paths


def get_diffusion_noise(
    size: Sequence[int], device: torch.device, train_config: TrainConfig
) -> Tensor:
    noise = torch.randn(size, device=device)

    if train_config.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += train_config.noise_offset * torch.randn(
            (size[0], size[1], 1, 1),
            device=device,
        )
    return noise


def get_diffusion_loss(
    diffusion_model: StableDiffusionModel,
    train_config: TrainConfig,
    model_input: Tensor,
    model_pred: Tensor,
    noise: Tensor,
    timesteps: Tensor,
) -> Tensor:
    noise_scheduler = diffusion_model.noise_scheduler
    prediction_type = noise_scheduler.config.prediction_type  # type: ignore

    if diffusion_model.config.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        noise_pred = noise_pred_uncond + diffusion_model.config.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # Get the target for loss depending on the prediction type
    if train_config.noise_scheduler_prediction_type is not None:
        # set prediction_type of scheduler if defined
        prediction_type = train_config.noise_scheduler_prediction_type
        noise_scheduler.register_to_config(prediction_type=prediction_type)

    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        # TODO: Update this with timesteps
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)  # type: ignore
    else:
        raise ValueError(f"Unknown prediction type {prediction_type}")

    if train_config.snr_gamma is None:
        loss = nn.functional.mse_loss(
            noise_pred.float(), target.float(), reduction="mean"
        )
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, train_config.snr_gamma * torch.ones_like(timesteps)],
            dim=1,
        ).min(dim=1)[0]
        if prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = nn.functional.mse_loss(
            noise_pred.float(), target.float(), reduction="none"
        )
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


@torch.no_grad()
def get_validation_metrics(
    accelerator: Accelerator,
    train_config: TrainConfig,
    train_state: TrainState,
    dataloader: DataLoader,
    diffusion_model: StableDiffusionModel,
    run_prefix: str,
):
    running_metrics = {
        name: torch.zeros(1, dtype=torch.float32)
        for name in diffusion_model.diffusion_metrics.keys()
    }
    running_count = 0

    for batch in dataloader:
        rgb = batch["rgb"].to(dtype=diffusion_model.dtype, device=accelerator.device)

        diffusion_output = diffusion_model.get_diffusion_output(batch)
        rgb_out = diffusion_output["rgb"]

        # Benchmark
        running_count += len(rgb)
        for metric_name, metric in diffusion_model.diffusion_metrics.items():
            values: Tensor = metric(rgb_out, rgb).detach().cpu()
            if len(values.shape) == 0:
                values = torch.tensor([values.item()])

            running_metrics[metric_name] += torch.sum(values)

            if train_config.use_debug_metrics:
                value_dict = {str(i): float(v) for i, v in enumerate(values)}
                accelerator.log(
                    {f"{run_prefix}_{metric_name}_{str(batch['meta'])}": value_dict},
                    step=train_state.global_step,
                )

    mean_metrics = {}
    for metric_name in diffusion_model.diffusion_metrics.keys():
        mean_metric = (running_metrics[metric_name] / running_count).item()
        mean_metrics[metric_name] = mean_metric

    return mean_metrics


@torch.no_grad()
def get_validation_renders(
    accelerator: Accelerator,
    train_config: TrainConfig,
    dataloader: DataLoader,
    diffusion_model: StableDiffusionModel,
):

    vis_rgb_gts = []
    vis_rgbs_noisy = []
    vis_metas = []
    vis_outs = []

    for batch in dataloader:
        rgb_gt = batch["rgb"].to(dtype=diffusion_model.dtype, device=accelerator.device)
        diffusion_output = diffusion_model.get_diffusion_output(batch)
        rgb_out = diffusion_output["rgb"]

        # Renders
        val_start_timestep = int(
            (1 - train_config.diffusion_config.noise_strength)
            * len(diffusion_model.noise_scheduler.timesteps)
        )

        rgb_noised = get_noised_img(
            rgb_gt,
            timestep=val_start_timestep,
            vae=diffusion_model.vae,
            img_processor=diffusion_model.img_processor,
            noise_scheduler=diffusion_model.noise_scheduler,
        )

        vis_rgb_gts.extend(list(rgb_gt.detach().cpu()))
        vis_outs.extend(list(rgb_out.detach().cpu()))
        vis_rgbs_noisy.extend(list(rgb_noised.detach().cpu()))
        vis_metas.extend(batch["meta"])

    return {
        f"rgb_gt": vis_rgb_gts,
        f"rgb_out": vis_outs,
        f"rgb_noisy": vis_rgbs_noisy,
        f"meta": vis_metas,
    }


@torch.no_grad()
def get_reference_outputs(
    accelerator: Accelerator,
    train_config: TrainConfig,
    train_state: TrainState,
    sd_model: StableDiffusionModel,
    dataloader: DataLoader,
    run_prefix: str,
):
    ref_nvs_running_metrics = {
        name: torch.zeros(1, dtype=torch.float32)
        for name in sd_model.diffusion_metrics.keys()
    }
    ref_running_metrics = {
        name: torch.zeros(1, dtype=torch.float32)
        for name in sd_model.diffusion_metrics.keys()
    }
    ref_running_count = 0
    ref_gts = []
    ref_nvs_outs = []
    ref_gen_outs = []
    ref_metas = []

    for batch in dataloader:
        ref_rgb = batch["rgb_nerf"].to(dtype=sd_model.dtype, device=accelerator.device)
        ref_gt = batch["rgb_gt"].to(dtype=sd_model.dtype, device=accelerator.device)

        diffusion_output = sd_model.get_diffusion_output(batch, rgb_key="rgb_nerf")
        ref_gen_out = diffusion_output["rgb"]

        ref_running_count += len(ref_rgb)

        ref_nvs_outs.extend(list(ref_rgb.detach().cpu()))
        ref_gen_outs.extend(list(ref_gen_out.detach().cpu()))
        ref_gts.extend(list(ref_gt.detach().cpu()))
        ref_metas.extend(batch["meta"])

        for metric_name, metric in sd_model.diffusion_metrics.items():
            values: Tensor = metric(ref_gen_out, ref_gt).detach().cpu()
            if len(values.shape) == 0:
                values = torch.tensor([values.item()])

            values_nvs: Tensor = metric(ref_rgb, ref_gt).detach().cpu()
            if len(values_nvs.shape) == 0:
                values_nvs = torch.tensor([values_nvs.item()])

            ref_running_metrics[metric_name] += torch.sum(values)
            ref_nvs_running_metrics[metric_name] += torch.sum(values_nvs)

            if train_config.use_debug_metrics:
                value_dict = {str(i): float(v) for i, v in enumerate(values)}
                value_dict_nvs = {str(i): float(v) for i, v in enumerate(values_nvs)}

                accelerator.log(
                    {
                        f"{run_prefix}_ref_nvs_{metric_name}_{str(batch['meta'])}": value_dict_nvs,
                        f"{run_prefix}_ref_{metric_name}_{str(batch['meta'])}": value_dict,
                    },
                    step=train_state.global_step,
                )

    mean_ref_metrics = {}
    mean_ref_nvs_metrics = {}
    for metric_name, metric in sd_model.diffusion_metrics.items():
        mean_ref_metrics[f"{run_prefix}_ref_{metric_name}"] = (
            ref_running_metrics[metric_name] / ref_running_count
        ).item()
        mean_ref_nvs_metrics[f"{run_prefix}_ref_nvs_{metric_name}"] = (
            ref_nvs_running_metrics[metric_name] / ref_running_count
        ).item()

    return (
        {
            "ref": mean_ref_metrics,
            "ref_nvs": mean_ref_nvs_metrics,
        },
        {
            f"ref_nvs_out": ref_nvs_outs,
            f"ref_gen_out": ref_gen_outs,
            f"ref_gt": ref_gts,
            f"ref_meta": ref_metas,
        },
    )


def prefix_keys(d, prefix):
    return {f"{prefix}_{n}": v for n, v in d.items()}


def pack_images_for_wandb(labels, all_imgs, metas):
    assert len(labels) == len(all_imgs)

    packed_imgs = {
        label: [
            wandb.Image(
                img,
                caption=str(meta),
            )
            for (img, meta) in zip(imgs, metas)
        ]
        for label, imgs in zip(labels, all_imgs)
    }
    return packed_imgs


def train_epoch(
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_config: TrainConfig,
    train_state: TrainState,
    dataloader: torch.utils.data.DataLoader,
    diffusion_model: StableDiffusionModel,
    params_to_optimize,
    progress_bar: tqdm.tqdm,
):
    diffusion_model.set_training(True)

    train_loss = 0.0

    for batch in dataloader:
        assert "rgb" in batch

        with accelerator.accumulate(diffusion_model.get_models_to_train()):
            rgb = batch["rgb"].to(
                dtype=diffusion_model.dtype, device=accelerator.device
            )
            rgb_gt = rgb

            with torch.no_grad():
                model_input = encode_img(
                    diffusion_model.img_processor,
                    diffusion_model.vae,
                    rgb,
                    accelerator.device,
                )

                noise = get_diffusion_noise(
                    model_input.shape, model_input.device, train_config
                )

                timesteps = get_random_timesteps(
                    train_config.train_noise_strength,
                    diffusion_model.noise_scheduler.config.num_train_timesteps,  # type: ignore
                    model_input.device,
                    model_input.size(0),
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = diffusion_model.noise_scheduler.add_noise(
                    cast(FloatTensor, model_input),
                    cast(FloatTensor, noise),
                    cast(IntTensor, timesteps),
                )

                prompt_hidden_state = encode_tokens(
                    diffusion_model.text_encoder,
                    batch["input_ids"],
                    use_cache=not diffusion_model.is_model_trained("text_encoder"),
                )

                if diffusion_model.config.do_classifier_free_guidance:
                    noisy_model_input = torch.cat(
                        (noisy_model_input, noisy_model_input)
                    )
                    prompt_hidden_state = torch.cat(
                        (prompt_hidden_state, prompt_hidden_state)
                    )

            unet_added_conditions = {}
            unet_kwargs = {}

            if diffusion_model.using_controlnet:

                conditioning = combine_conditioning_info(
                    batch, diffusion_model.conditioning_signals
                ).to(noisy_model_input.device)

                if diffusion_model.config.do_classifier_free_guidance:
                    conditioning = torch.cat((conditioning, conditioning))

                down_block_res_samples, mid_block_res_sample = (
                    diffusion_model.controlnet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states=prompt_hidden_state,
                        controlnet_cond=conditioning,
                        return_dict=False,
                    )
                )

                unet_kwargs["down_block_additional_residuals"] = [
                    sample.to(dtype=diffusion_model.dtype)
                    for sample in down_block_res_samples
                ]
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample.to(
                    dtype=diffusion_model.dtype
                )

            model_pred = diffusion_model.unet(
                noisy_model_input,
                timesteps,
                prompt_hidden_state,
                added_cond_kwargs=unet_added_conditions,
                **unet_kwargs,
            ).sample

            loss = get_diffusion_loss(
                diffusion_model, train_config, model_input, model_pred, noise, timesteps
            )

            if train_config.use_recreation_loss:
                raise NotImplementedError
                # TODO: Completely change this
                # DENOISE LATENT
                # DECODE IMAGE
                rec_loss = (
                    torch.mean(
                        train_config.rec_loss_strength
                        * nn.functional.mse_loss(pred_rgb, rgb_gt, reduce=None)
                        * (1 - (timesteps + 1) / noise_scheduler.num_train_timesteps)
                    )
                    ** 2
                )
                loss += rec_loss

            if train_config.use_debug_metrics and "meta" in batch:
                for meta in batch["meta"]:
                    accelerator.log(
                        {"loss_for_" + str(meta): loss},
                        step=train_state.global_step,
                    )

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = (
                cast(
                    Tensor,
                    accelerator.gather(loss.repeat(train_config.train_batch_size)),
                ).mean()
                / train_config.gradient_accumulation_steps
            )

            train_loss += avg_loss.item()
            train_state.loss_history.append(avg_loss.item())

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    params_to_optimize, train_config.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            train_state.global_step += 1

            loss_window = 100
            running_loss = np.mean(train_state.loss_history[-loss_window:])

            accelerator.log({"train_loss": train_loss}, step=train_state.global_step)
            accelerator.log(
                {"train_loss (100 steps)": running_loss},
                step=train_state.global_step,
            )

            logs = {
                "epoch": train_state.epoch,
                "train_loss": train_loss,
                f"running_loss ({loss_window})": running_loss,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            train_loss = 0.0

    train_state.epoch += 1


@torch.no_grad()
def validate_model(
    accelerator: Accelerator,
    train_config: TrainConfig,
    train_state: TrainState,
    dataloader: torch.utils.data.DataLoader,
    vis_dataloader: Optional[torch.utils.data.DataLoader],
    ref_dataloader: Optional[torch.utils.data.DataLoader],
    diffusion_model: StableDiffusionModel,
    run_prefix: str,
) -> None:
    diffusion_model.set_training(False)

    val_metrics = get_validation_metrics(
        accelerator,
        train_config,
        train_state,
        dataloader,
        diffusion_model,
        run_prefix,
    )

    val_renders = (
        get_validation_renders(
            accelerator,
            train_config,
            vis_dataloader,
            diffusion_model,
        )
        if vis_dataloader is not None
        else None
    )

    ref_metrics, ref_renders = (
        get_reference_outputs(
            accelerator,
            train_config,
            train_state,
            diffusion_model,
            ref_dataloader,
            run_prefix,
        )
        if ref_dataloader is not None
        else (None, None)
    )

    for metric_name, metric in val_metrics.items():
        if (
            metric_name == train_config.checkpoint_metric
            and metric > train_state.best_metric
        ):
            train_state.best_metric = metric

    accelerator.log(
        prefix_keys(val_metrics, run_prefix),
        step=train_state.global_step,
    )

    if ref_metrics:
        accelerator.log(
            prefix_keys(ref_metrics["ref"], run_prefix),
            step=train_state.global_step,
        )

        accelerator.log(
            prefix_keys(ref_metrics["ref_nvs"], run_prefix),
            step=train_state.global_step,
        )

    imgs_to_log = {}
    if val_renders:
        imgs_to_log.update(
            pack_images_for_wandb(
                [
                    f"{run_prefix}_ground_truth",
                    f"{run_prefix}_diffusion_output",
                    f"noisy_{run_prefix}_ground_truth",
                ],
                [
                    val_renders["rgb_gt"],
                    val_renders["rgb_out"],
                    val_renders["rgb_noisy"],
                ],
                val_renders["meta"],
            )
        )
    if ref_renders:
        imgs_to_log.update(
            pack_images_for_wandb(
                [
                    f"{run_prefix}_reference_ground_truth",
                    f"{run_prefix}_reference_nvs_output",
                    f"{run_prefix}_reference_diffusion_output",
                ],
                [
                    ref_renders["ref_gt"],
                    ref_renders["ref_nvs_out"],
                    ref_renders["ref_gen_out"],
                ],
                ref_renders["ref_meta"],
            )
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            raise NotImplementedError

        elif tracker.name == "wandb":
            tracker.log_images(
                imgs_to_log,
                step=train_state.global_step,
            )


def setup_accelerator(train_config: TrainConfig) -> Accelerator:

    (train_config.logging_dir / "accelerator").mkdir(exist_ok=True, parents=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=(
            train_config.diffusion_config.dtype
            if train_config.diffusion_config.dtype in LOWER_DTYPES
            else "no"
        ),
        log_with=list(filter(lambda s: s != "", train_config.loggers)),
        project_config=ProjectConfiguration(
            project_dir=str(train_config.output_dir),
            logging_dir=str(train_config.logging_dir / "accelerator"),
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logger.info(f"Launching script under id: {train_config.job_id}")
    logger.info(
        f"Number of cuda detected devices: {torch.cuda.device_count()}, Using device: {accelerator.device}, distributed: {accelerator.distributed_type}"
    )

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

    if train_config.seed is not None:
        set_seed(train_config.seed)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # Note: this applies to the A100
    if train_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if (
        torch.backends.mps.is_available()
        and train_config.diffusion_config.dtype == "bf16"
    ):
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if accelerator.is_main_process:
        if not train_config.output_dir.exists():
            train_config.output_dir.mkdir(exist_ok=True, parents=True)
            (train_config.output_dir / train_config.job_id).mkdir(exist_ok=True)
            train_config.logging_dir.mkdir(exist_ok=True)

        if train_config.push_to_hub:
            raise NotImplementedError

    if "wandb" in train_config.loggers:
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

        if train_config.hub_token is not None:
            raise ValueError(
                "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        import wandb

        if accelerator.is_local_main_process:
            (train_config.logging_dir / "wandb").mkdir(exist_ok=True, parents=True)
            wandb.init(
                project=train_config.wandb_project or os.getenv("WANDB_PROJECT"),
                entity=train_config.wandb_entity or os.getenv("WANDB_ENTITY"),
                dir=train_config.logging_dir / "wandb",
                group=train_config.wandb_group or os.getenv("WANDB_GROUP"),
                reinit=True,
                config=train_config.model_dump(),
            )

    return accelerator


def prepare_models_with_accelerator(
    train_config: TrainConfig,
    train_state: TrainState,
    diffusion_model: StableDiffusionModel,
    accelerator: Accelerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.Optimizer]:
    num_epochs = train_config.n_epochs
    num_processes = accelerator.num_processes
    lr_warmup_steps = train_config.lr_warmup_steps
    len_dataloader = len(dataloader)
    gradient_accumulation_steps = train_config.gradient_accumulation_steps

    num_warmup_steps = lr_warmup_steps * num_processes
    len_dataloader_after_sharding = math.ceil(len_dataloader / num_processes)
    num_update_steps_per_epoch = math.ceil(
        len_dataloader_after_sharding / gradient_accumulation_steps
    )
    num_steps_for_scheduler = num_epochs * num_update_steps_per_epoch * num_processes

    lr_scheduler = get_scheduler(
        name=train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps_for_scheduler,
        **train_config.lr_scheduler_kwargs,
    )

    # Override internal models of diffusion model with the ones prepared with accelerator to ensure correct device placement.
    optimizer, train_dataloader, lr_scheduler, *trainable_models = accelerator.prepare(
        optimizer,
        dataloader,
        lr_scheduler,
        *diffusion_model.get_models_to_train(),
    )
    lr_scheduler = cast(torch.optim.lr_scheduler.LambdaLR, lr_scheduler)
    optimizer = cast(torch.optim.Optimizer, optimizer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(len_dataloader / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    if num_steps_for_scheduler != max_train_steps * num_processes:
        logger.warning(
            f"The length of the 'train_dataloader' after 'accelerator.prepare' ({num_epochs}) does not match "
            f"the expected length ({len_dataloader_after_sharding}) when the learning rate scheduler was created. "
            f"This inconsistency may result in the learning rate scheduler not functioning properly."
        )

    diffusion_model.set_models(
        train_config.diffusion_config.models_to_train_lora, trainable_models
    )

    train_state.num_update_steps_per_epoch = num_update_steps_per_epoch
    train_state.max_train_steps = max_train_steps

    return lr_scheduler, optimizer


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    if not torch.cuda.is_available():
        logging.warning(
            f"CUDA not detected. Running on CPU. The code is not supported for CPU and will most likely give incorrect results. Proceed with caution."
        )

    train_config = TrainConfig()
    train_state = TrainState()

    accelerator = setup_accelerator(train_config)

    setup_cache(train_config.cache_dir)

    diffusion_model = StableDiffusionModel(
        train_config.diffusion_config,
        device=accelerator.device,
    )
    diffusion_model.set_gradient_checkpointing(
        train_config.enable_gradient_checkpointing
    )

    accelerator.register_save_state_pre_hook(
        functools.partial(
            save_model_hook,
            accelerator=accelerator,
            diffusion_model=diffusion_model,
            train_config=train_config,
            train_state=train_state,
        )
    )
    accelerator.register_load_state_pre_hook(
        functools.partial(
            load_model_hook,
            accelerator=accelerator,
            diffusion_model=diffusion_model,
            train_config=train_config,
        )
    )

    shared_dataloader_kwargs = {
        "batch_size": train_config.train_batch_size,
        "num_workers": train_config.dataloader_num_workers,
        "pin_memory": train_config.pin_memory,
    }
    train_dataset, train_dataloader = train_config.dataset_configs.make_dataset_loader(
        "train",
        **shared_dataloader_kwargs,
    )
    val_dataset, val_dataloader = train_config.dataset_configs.make_dataset_loader(
        "val",
        **shared_dataloader_kwargs,
    )
    _, render_dataloader = train_config.dataset_configs.make_dataset_loader(
        "render", **shared_dataloader_kwargs
    )
    _, nerf_out_dataloader = train_config.dataset_configs.make_dataset_loader(
        "nerf_out", **shared_dataloader_kwargs
    )
    assert train_dataset and train_dataloader
    assert val_dataset and val_dataloader

    # ==========================
    # ===   Setup training   ===
    # ==========================
    params_to_optimize = list(
        filter(
            lambda p: p.requires_grad,
            it.chain(
                *(model.parameters() for model in diffusion_model.get_models_to_train())
            ),
        )
    )

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=(
            (
                train_config.learning_rate
                * train_config.gradient_accumulation_steps
                * train_config.train_batch_size
                * accelerator.num_processes
            )
            if train_config.scale_lr
            else train_config.learning_rate
        ),
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        weight_decay=train_config.adam_weight_decay,
        eps=train_config.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    lr_scheduler, optimizer = prepare_models_with_accelerator(
        train_config,
        train_state,
        diffusion_model,
        accelerator,
        train_dataloader,
        optimizer,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            train_config.wandb_group, config=train_config.model_dump()
        )

    # ===================
    # === Train model ===
    # ===================

    logger.info("***** Running training *****")
    logger.info(f"  Num training samples = {len(train_dataset)}")
    logger.info(f"  Num validation samples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {train_state.max_epoch}")
    logger.info(
        f"  Instantaneous batch size per device = {train_config.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {train_state.max_epoch * train_state.num_update_steps_per_epoch * accelerator.num_processes}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {train_state.max_train_steps}")
    logger.info(f"Configuration: {train_config}")

    if train_config.checkpoint_path:
        resume_from_checkpoint(accelerator, train_config, train_state)

    progress_bar = tqdm.tqdm(
        range(0, train_state.max_train_steps),
        initial=train_state.global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    while train_state.epoch < train_config.n_epochs:
        train_epoch(
            accelerator,
            optimizer,
            lr_scheduler,
            train_config,
            train_state,
            train_dataloader,
            diffusion_model,
            params_to_optimize,
            progress_bar,
        )
        torch.cuda.empty_cache()

        if train_state.epoch % train_config.val_freq == 0:
            validate_model(
                accelerator,
                train_config,
                train_state,
                val_dataloader,
                render_dataloader,
                nerf_out_dataloader,
                diffusion_model,
                "val",
            )
        torch.cuda.empty_cache()

        # if accelerator.is_main_process and (
        #    train_config.global_step % train_config.checkpointing_steps == 0
        # ):
        if accelerator.is_main_process:
            if train_config.checkpoint_strategy == "best" and is_metric_improved(
                train_config.checkpoint_metric,
                train_state.best_saved_metric,
                train_state.best_metric,
            ):
                save_checkpoint(accelerator, train_config, train_state)
                train_state.best_saved_metric = train_state.best_metric

            elif (
                train_config.checkpoint_strategy == "latest"
                and train_state.global_step % train_config.checkpointing_steps == 0
            ):
                save_checkpoint(accelerator, train_config, train_state)

        if (
            train_state.global_step >= train_state.max_train_steps
            or train_state.epoch > train_state.max_epoch
        ):
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Final inference
        validate_model(
            accelerator,
            train_config,
            train_state,
            val_dataloader,
            render_dataloader,
            nerf_out_dataloader,
            diffusion_model,
            "test",
        )

        # Save the lora layers
        save_lora_weights(accelerator, diffusion_model, train_config, "final_weights")

    accelerator.end_training()


if __name__ == "__main__":
    main()
