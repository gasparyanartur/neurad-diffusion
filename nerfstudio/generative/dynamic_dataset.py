import typing
from typing_extensions import Annotated
from pydantic import BaseModel, StringConstraints
from typing import Any, Dict, List, Set, Type, Union, Optional, Tuple, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections.abc import Iterable, Generator, Callable
import os
from pathlib import Path
import json
import yaml
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchvision

from transformers import CLIPTokenizer, AutoTokenizer

from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
from nerfstudio.data.dataparsers.pandaset_dataparser import AVAILABLE_CAMERAS
from nerfstudio.model_components.ray_generators import RayGenerator

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as tvtf
import torchvision
import logging

from nerfstudio.generative.utils import (
    DTYPE_CONVERSION,
    get_env,
    set_env,
    set_if_no_key,
)
import pyquaternion


norm_img_pipeline = tvtf.Compose([tvtf.ConvertImageDtype(torch.float32)])
norm_img_crop_pipeline = tvtf.Compose(
    [
        tvtf.ConvertImageDtype(torch.float32),
        tvtf.CenterCrop((1024, 1024)),
        tvtf.Resize((512, 512), antialias=True),
    ]
)
norm_img_rand_crop_pipeline = tvtf.Compose(
    [
        tvtf.ConvertImageDtype(torch.float32),
        tvtf.RandomCrop((1024, 1024)),
        tvtf.Resize((512, 512), antialias=True),
    ]
)


def make_img_tf_pipe(
    crop_size: Optional[Tuple[int, int]] = (1024, 1024),
    resize_factor: Optional[float] = 2,
    crop_type: str = "center",
    dtype: Optional[torch.dtype] = torch.float32,
    device: Union[str, torch.device] = "cpu",
):
    pipe = []

    if dtype:
        pipe.append(tvtf.ConvertImageDtype(dtype))

    if crop_size:
        if crop_type == "random":
            pipe.append(tvtf.RandomCrop(crop_size))

        elif crop_type == "center":
            pipe.append(tvtf.CenterCrop(crop_size))

        elif crop_type == "none":
            ...

        else:
            logging.warning(
                f"Could not recognize crop type `{crop_type}`, skipping cropping."
            )

    if resize_factor:
        if not crop_size:
            logging.warning(
                f"Could not set resize factor when `crop_size` is set to `None`."
            )

        else:
            pipe.append(
                tvtf.Resize(
                    (
                        int(crop_size[0] // resize_factor),
                        int(crop_size[1] // resize_factor),
                    ),
                    antialias=True,
                )
            )

    return tvtf.Compose(pipe).to(device=device)


DATA_SUFFIXES: Dict[Tuple[str, str], str] = {
    ("rgb", "pandaset"): ".jpg",
    ("rgb", "neurad"): ".jpg",
    ("ref-rgb", "pandaset"): ".jpg",
    ("gt-rgb", "neurad"): ".jpg",
    ("lidar", "pandaset"): ".pkl.gz",
    ("lidar", "neurad"): ".pkl.gz",
    ("pose", "pandaset"): ".json",
    ("intrinsics", "pandaset"): ".json",
    ("timestamp", "pandaset"): ".json",
    ("camera", "pandaset"): "",
}
RANGE_SEP = ":"
CN_SIGNAL_PATTERN = re.compile(
    r"cn_(?P<cn_type>\w+)_(?P<num_channels>\d+)_(?P<camera>\w+)"
)


def get_dataset_from_path(path: Path) -> str:
    return path.stem


def save_yaml(path: Path, data: Dict[str, Any]):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def setup_cache(cache_dir: Path):
    set_env("HF_HUB_CACHE", cache_dir / "hf")  # Huggingface cache dir
    set_env("MPLCONFIGDIR", cache_dir / "mpl")  # Matplotlib cache dir


def img_float_to_img(img: Tensor):
    img = img * 255
    img = img.to(device=img.device, dtype=torch.uint8)
    return img


def sort_paths_numerically(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda path: int(path.stem))


def load_json(path):
    with open(path, "rb") as f:
        return json.load(f)


def load_img_paths_from_dir(dir_path: Path):
    img_paths = list(sorted(dir_path.glob("*.jpg")))
    return img_paths


def read_image(
    img_path: Path,
    tf_pipeline: tvtf.Compose = norm_img_pipeline,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    # TODO: Perform random crop directly in here, return extra information
    img = torchvision.io.read_image(str(img_path)).to(device=device)
    img = tf_pipeline(img)

    return img


def save_image(save_path: Path, img: Tensor, jpg_quality: int = 100) -> None:
    img = img.detach().cpu()
    img = img.squeeze()

    if torch.is_floating_point(img):
        img = img_float_to_img(img)

    torchvision.io.write_jpeg(img, str(save_path), quality=jpg_quality)


def save_json(save_path: Path, data: Dict[str, Any]) -> None:
    with open(save_path, "w") as f:
        json.dump(data, f)


def load_img_if_path(img: Union[str, Path, Tensor]) -> Tensor:
    if isinstance(img, str):
        img = Path(img)

    if isinstance(img, Path):
        img = read_image(img)

    return img


def iter_numeric_names(
    start_name: Union[str, int], end_name: Union[str, int], fixed_len: int = 2
):
    start_name = int(start_name)
    end_name = int(end_name)

    for i in range(start_name, end_name + 1):
        num = str(i)
        if fixed_len:
            num = num.rjust(fixed_len, "0")

        yield num


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


SampleNameString = Annotated[str, StringConstraints(pattern=re.compile(r"\d{2}"))]
SceneNameString = Annotated[str, StringConstraints(pattern=re.compile(r"\d{3}"))]
SplitNameString = Annotated[str, StringConstraints(pattern=re.compile(r"\w+"))]
DatasetNameString = Annotated[str, StringConstraints(pattern=re.compile(r"[\w-]+"))]


class DatasetTreeSample(BaseModel):
    sample_name: SampleNameString


class DatasetTreeScene(BaseModel, ABC):
    scene_name: SceneNameString


class DatasetTreeSceneList(DatasetTreeScene):
    samples: List[DatasetTreeSample]


class DatasetTreeSceneRange(DatasetTreeScene):
    start_scene: Optional[SampleNameString]
    end_scene: Optional[SampleNameString]
    skip_scene: Optional[int]


class DatasetTreeSplit(BaseModel):
    split_name: SplitNameString
    scenes: Dict[SceneNameString, DatasetTreeScene]


class DatasetTree(BaseModel):
    dataset_name: DatasetNameString
    splits: Dict[SplitNameString, DatasetTreeSplit]

    @classmethod
    def single_split_single_scene(
        cls,
        dataset_name: DatasetNameString,
        split_name: SplitNameString,
        scene_name: SceneNameString,
        sample_names: List[SampleNameString],
    ) -> "DatasetTree":
        return DatasetTree(
            dataset_name=dataset_name,
            splits={
                split_name: DatasetTreeSplit(
                    split_name=split_name,
                    scenes={
                        scene_name: DatasetTreeSceneList(
                            scene_name=scene_name,
                            samples=[
                                DatasetTreeSample(sample_name=sample_name)
                                for sample_name in sample_names
                            ],
                        )
                    },
                )
            },
        )


def _pandaset_pose_to_matrix(pandaset_pose):
    pose = torch.eye(4)
    quaternion = np.array(
        [
            pandaset_pose["heading"]["w"],
            pandaset_pose["heading"]["x"],
            pandaset_pose["heading"]["y"],
            pandaset_pose["heading"]["z"],
        ]
    )
    pose[:3, :3] = torch.from_numpy(
        pyquaternion.Quaternion(quaternion).rotation_matrix @ OPENCV_TO_NERFSTUDIO
    )
    translation = torch.tensor(
        [
            pandaset_pose["position"]["x"],
            pandaset_pose["position"]["y"],
            pandaset_pose["position"]["z"],
        ]
    )
    pose[:3, 3] = translation
    return pose


def create_cameras_for_sequence(
    poses: Tensor,
    timestamps: Tensor,
    intrinsics: Tensor,
    scene_idxs: Tensor,
    img_width=1920,
    img_height=1080,
):
    assert len(poses) == len(timestamps) == len(intrinsics) == len(scene_idxs)

    timestamps = timestamps.to(dtype=torch.float64)  # need higher precision
    cameras = Cameras(
        fx=intrinsics[:, 0, 0],
        fy=intrinsics[:, 1, 1],
        cx=intrinsics[:, 0, 2],
        cy=intrinsics[:, 1, 2],
        height=img_height,
        width=img_width,
        camera_to_worlds=(poses[:, :3, :4]),
        camera_type=CameraType.PERSPECTIVE,
        times=torch.tensor(timestamps),
        metadata={"sensor_idxs": scene_idxs},
    )
    return cameras


def crop_to_ray_idxs(cam_idxs, crop_top_left, crop_size) -> Tensor:
    is_batched = len(cam_idxs) > 1
    if is_batched:
        raise NotImplementedError

    device = crop_top_left.device
    C = cam_idxs.size(-1)
    H = crop_size[..., -2]
    W = crop_size[..., -1]

    idxs = torch.cartesian_prod(
        torch.arange(C, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
    )
    idxs[..., 0] = cam_idxs
    idxs[..., 1:] += crop_top_left[..., :]

    return idxs


def load_cameras(
    sample_config: "SampleConfig",
    sample_infos: List["SampleInfo"],
    pose_getter: "PoseDataGetter",
    timestamp_getter: "TimestampDataGetter",
    intrinsics_getter: "IntrinsicsDataGetter",
    img_width: int = 1920,
    img_height: int = 1080,
) -> Dict[str, Cameras]:
    # dataset / scene / sample
    cameras = {}

    poses = {}
    timestamps = {}
    intrinsics = {}
    sensor_idxs = {}

    # Assume const dataset
    dataset = sample_infos[0].dataset
    assert (info.dataset == dataset for info in sample_infos)

    unique_scenes = {info.scene for info in sample_infos}
    scenes = {
        scene: [
            info
            for info in sample_infos
            if info.scene == scene and info.dataset == dataset
        ]
        for scene in unique_scenes
    }

    for scene, samples in scenes.items():
        args = [construct_sample_argument(sample_config, info) for info in samples]

        poses = torch.stack([pose_getter.get_data(arg) for arg in args])
        timestamps = torch.tensor([timestamp_getter.get_data(arg) for arg in args])
        intrinsics = torch.stack([intrinsics_getter.get_data(arg) for arg in args])
        sensor_idxs = torch.tensor([int(info.sample) for info in samples])

        cameras[scene] = create_cameras_for_sequence(
            poses, timestamps, intrinsics, sensor_idxs, img_width, img_height
        )

    return cameras


@dataclass
class SampleInfo:
    dataset: str
    scene: str
    sample: str
    split: str

    def __str__(self) -> str:
        return f"{self.dataset} - {self.scene} - {self.sample}"

    def __cmp__(self, other: "SampleInfo") -> int:
        split_a = _SPLIT_SCORE[self.split]
        split_b = _SPLIT_SCORE[other.split]
        if split_a > split_b:
            return 1
        elif split_a < split_b:
            return -1

        scene_a = int(self.scene)
        scene_b = int(other.scene)
        if scene_a > scene_b:
            return 1
        elif scene_a < scene_b:
            return -1

        sample_a = int(self.sample)
        sample_b = int(other.sample)
        if sample_a > sample_b:
            return 1
        elif sample_a < sample_b:
            return -1

        return 0

    def __lt__(self, other: "SampleInfo") -> bool:
        return self.__cmp__(other) == -1

    def __gt__(self, other: "SampleInfo") -> bool:
        return self.__cmp__(other) == 1

    def __eq__(self, other: "SampleInfo") -> bool:
        return self.__cmp__(other) == 0


class DataSpec(BaseModel):
    name: str

    @abstractmethod
    def get_getter_class(self) -> Type["DataGetter"]:
        raise NotImplementedError

    def create_getter(
        self, info_getter: "InfoGetter", sample_config: "SampleConfig"
    ) -> "DataGetter":
        return self.get_getter_class()(info_getter, self, sample_config)


class CameraDataSpec(DataSpec):
    camera: str
    shift: str

    def get_getter_class(self) -> Type["DataGetter"]:
        return CameraDataGetter


class CaptureDataSpec(DataSpec):
    camera: str = "front"
    shift: str = "0m"

    def get_getter_class(self) -> Type["DataGetter"]:
        return RGBDataGetter


class LidarDataSpec(DataSpec):
    shift: str

    def get_getter_class(self) -> Type["DataGetter"]:
        return LidarDataGetter


class RgbDataSpec(CaptureDataSpec):
    name: str = "rgb"
    num_channels: int = 3
    width: int = 1920
    height: int = 1080
    dtype: str = "fp32"
    normalize: bool = True

    def get_getter_class(self) -> Type["DataGetter"]:
        return RGBDataGetter


class NerfOutputSpec(RgbDataSpec):
    nerf_output_path: str = "data/nerf_outputs"
    data_name: str = "rgb"

    def get_getter_class(self) -> Type["DataGetter"]:
        return NerfOutputDataGetter


class PromptDataSpec(DataSpec):
    name: str = "input_ids"
    prompt: str = ""
    model_id: str = "stabilityai/stable-diffusion-2-1"
    subfolder: str = "tokenizer"
    revision: str = "main"

    def get_getter_class(self) -> Type["DataGetter"]:
        return PromptDataGetter


class PoseDataSpec(CaptureDataSpec):
    def get_getter_class(self) -> Type["DataGetter"]:
        return PoseDataGetter


class IntrinsicsDataSpec(CaptureDataSpec):
    def get_getter_class(self) -> Type["DataGetter"]:
        return IntrinsicsDataGetter


class TimestampDataSpec(CaptureDataSpec):
    def get_getter_class(self) -> Type["DataGetter"]:
        return TimestampDataGetter


class RayDataSpec(CameraDataSpec):
    def get_getter_class(self) -> Type["DataGetter"]:
        return RayDataGetter


_SPLIT_SCORE = {"train": 0, "validation": 1, "test": 2}


def get_cam_idxs(infos: Iterable[SampleInfo]) -> Dict[str, Dict[str, int]]:
    cam_to_idx = {}
    for info in infos:
        if info.scene not in cam_to_idx:
            cam_to_idx[info.scene] = {}

        cam_to_idx[info.scene][info.sample] = len(cam_to_idx[info.scene])
    return cam_to_idx


class InfoGetter(ABC):
    def __init__(
        self, dataset_name: str, dataset_path: Path, dataset_tree: DatasetTree
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_tree = dataset_tree
        self.sample_infos: List[SampleInfo] = []

        self._parse_tree()

    @abstractmethod
    def get_scenes(self, split: str):
        raise NotImplementedError

    @abstractmethod
    def get_samples(
        self,
        scene: str,
        split: str,
        spec: Optional[DataSpec] = None,
    ) -> List[SampleInfo]:
        raise NotImplementedError

    @abstractmethod
    def get_path(self, info: SampleInfo, spec: DataSpec):
        raise NotImplementedError

    @abstractmethod
    def get_suffix(self, spec: DataSpec) -> str:
        raise NotImplementedError

    def _parse_tree(self) -> None:
        self.sample_infos = []

        for split_name, split_tree in self.data_tree.splits.items():
            existing_scenes = self.get_scenes(split_name)

            for scene_name, sample_tree in split_tree.scenes.items():
                if scene_name not in existing_scenes:
                    continue

                if isinstance(sample_tree, DatasetTreeSceneRange):
                    samples = sorted(self.get_samples(scene_name, split_name))

                    start_range = (
                        int(sample_tree.start_scene) if sample_tree.start_scene else 0
                    )
                    end_range = (
                        int(sample_tree.end_scene)
                        if sample_tree.end_scene
                        else len(samples)
                    )
                    skip_range = sample_tree.skip_scene or 1

                    sample_infos = samples[start_range:end_range:skip_range]

                elif isinstance(sample_tree, DatasetTreeSceneList):
                    if not sample_tree:
                        continue

                    assert isinstance(sample_tree, list)
                    assert isinstance(sample_tree[0], str)

                    sample_infos = [
                        SampleInfo(
                            self.data_tree.dataset_name,
                            scene_name,
                            sample.sample_name,
                            split_name,
                        )
                        for sample in sample_tree.samples
                    ]

                else:
                    raise NotImplementedError

                self.sample_infos.extend(sample_infos)


class PandasetInfoGetter(InfoGetter):
    def __init__(self, dataset_path: Path, data_tree: Dict[str, Any]) -> None:
        super().__init__("pandaset", dataset_path, data_tree)

    def get_scenes(self, split: str) -> Set[str]:
        return set(path.stem for path in self.dataset_path.iterdir())

    def get_suffix(self, spec: DataSpec) -> str:
        if isinstance(spec, RgbDataSpec):
            return ".jpg"
        elif isinstance(spec, LidarDataSpec):
            return ".pkl.gz"
        elif isinstance(spec, NerfOutputSpec):
            return ".jpg"
        elif isinstance(spec, CameraDataSpec):
            return ""
        else:
            raise NotImplementedError

    def get_samples(
        self,
        scene: str,
        split: str,
        spec: Optional[DataSpec] = None,
    ) -> List[str]:
        if isinstance(spec, LidarDataSpec):
            sample_dir = self.dataset_path / scene / "lidar"
            suffix = self.get_suffix(spec)

        elif isinstance(spec, NerfOutputSpec):
            sample_dir = (
                Path(spec.nerf_output_path)
                / scene
                / spec.camera
                / spec.shift
                / split
                / spec.data_name
            )
            suffix = self.get_suffix(spec)

        elif isinstance(spec, CameraDataSpec):
            sample_dir = self.dataset_path / scene / "camera" / spec.camera
            suffix = self.get_suffix(spec)

        else:
            sample_dir = self.dataset_path / scene / "camera" / "front"
            suffix = ""

        return [path.stem for path in sample_dir.glob(f"*{suffix}")]

    def get_path(self, info: SampleInfo, spec: DataSpec) -> Path:
        if isinstance(spec, LidarDataSpec):
            sample_path = self.dataset_path / info.scene / "lidar" / info.sample

        elif isinstance(spec, NerfOutputSpec):
            sample_path = (
                Path(spec.nerf_output_path)
                / info.scene
                / spec.camera
                / spec.shift
                / info.split
                / spec.data_name
                / info.sample
            )

        elif isinstance(spec, RgbDataSpec):
            sample_path = (
                self.dataset_path / info.scene / "camera" / spec.camera / info.sample
            )

        elif isinstance(spec, PoseDataSpec):
            sample_path = (
                self.dataset_path / info.scene / "camera" / spec.camera / "poses"
            )

        elif isinstance(spec, IntrinsicsDataSpec):
            sample_path = (
                self.dataset_path / info.scene / "camera" / spec.camera / "intrinsics"
            )

        elif isinstance(spec, CaptureDataSpec):
            sample_path = (
                self.dataset_path / info.scene / "camera" / spec.camera / info.sample
            )

        else:
            sample_path = (
                self.dataset_path / info.scene / "camera" / "front" / info.sample
            )

        sample_path = sample_path.with_suffix(self.get_suffix(spec))
        return sample_path


class SampleArgument(BaseModel):
    sample_info: SampleInfo
    crop_size: Optional[Tuple[int, int]] = None
    crop_top_left: Optional[Tuple[int, int]] = None
    do_flip: Optional[bool] = None


class DataGetter(ABC):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: DataSpec,
        sample_config: "SampleConfig",
    ) -> None:
        super().__init__()
        self.info_getter = info_getter
        self.data_spec = data_spec
        self.sample_config = sample_config

    def get_data_path(self, info: SampleInfo) -> Path:
        return self.info_getter.get_path(info, self.data_spec)

    @abstractmethod
    def get_data(self, args: SampleArgument) -> Any:
        raise NotImplementedError


class PromptDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: PromptDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)
        self.data_spec = data_spec
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.data_spec.model_id,
            revision=self.data_spec.revision,
            subfolder=self.data_spec.subfolder,
        )

    def get_data(self, args: SampleArgument) -> Tensor:
        from nerfstudio.generative.diffusion_model import tokenize_prompt

        return tokenize_prompt(self.tokenizer, self.data_spec.prompt)


class RGBDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: RgbDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)
        self.data_spec = data_spec

        self.transform = tvtf.Compose(
            [
                (
                    tvtf.ConvertImageDtype(DTYPE_CONVERSION[data_spec.dtype])
                    if data_spec.normalize
                    else tvtf.ToDtype(DTYPE_CONVERSION[data_spec.dtype])
                ),
                tvtf.Resize(
                    (sample_config.img_height, sample_config.img_width), antialias=True
                ),
            ]
        )

    def get_data(self, args: SampleArgument) -> Tensor:
        rgb_path = self.get_data_path(args.sample_info)
        rgb = read_image(rgb_path, self.transform)

        if (args.crop_top_left and not args.crop_size) or (
            args.crop_size and not args.crop_top_left
        ):
            raise ValueError(
                f"Both crop_top_left and crop_size must be set if either is set."
            )

        if args.crop_top_left and args.crop_size:
            rgb = tvtf.functional.crop(rgb, *args.crop_top_left, *args.crop_size)

        rgb = tvtf.functional.hflip(rgb) if args.do_flip else rgb

        return rgb


class LidarDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: LidarDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)
        self.data_spec = data_spec

    def get_data(self, args: SampleArgument) -> Tensor:
        path = self.get_data_path(args.sample_info)
        data = pd.read_pickle(path)

        return data


class PoseDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: PoseDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)

    def get_data(self, args: SampleArgument) -> Tensor:
        file_path = self.get_data_path(args.sample_info)
        poses = load_json(file_path)

        pose = poses[int(args.sample_info.sample)]
        pose = _pandaset_pose_to_matrix(pose)

        return pose.to(dtype=torch.float32, device=pose.device)


class IntrinsicsDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: IntrinsicsDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)

    def get_data(self, args: SampleArgument) -> Tensor:
        file_path = self.get_data_path(args.sample_info)

        data = load_json(file_path)
        intrinsics = torch.tensor(
            [
                [data["fx"], 0, data["cx"]],
                [0, data["fy"], data["cy"]],
                [0, 0, 1],
            ]
        )
        return intrinsics


class TimestampDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: TimestampDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)

    def get_data(self, args: SampleArgument) -> float:
        file_path = self.get_data_path(args.sample_info)

        all_timestamps = load_json(file_path)
        timestamp = all_timestamps[int(args.sample_info.sample)]
        return timestamp


class NerfOutputDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: NerfOutputSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)

        tf_layers = [
            (
                tvtf.ConvertImageDtype(DTYPE_CONVERSION[data_spec.dtype])
                if data_spec.normalize
                else tvtf.ToDtype(DTYPE_CONVERSION[data_spec.dtype])
            ),
            tvtf.Resize(
                (sample_config.img_height, sample_config.img_width),
                antialias=True,
            ),
        ]
        if sample_config.crop_size:
            tf_layers.append(tvtf.CenterCrop(sample_config.crop_size))

        self.transform = tvtf.Compose(tf_layers)

    def get_data(self, args: SampleArgument) -> Tensor:
        path = self.get_data_path(args.sample_info)
        data = read_image(path, self.transform)

        return data


class CameraDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: CameraDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)

        if info_getter.dataset_name == "pandaset" and data_spec.camera == "back":
            raise NotImplementedError(
                "Back camera not supported yet for backcamera (needs 1080-250 height)."
            )

        self.cam_to_idx = get_cam_idxs(self.info_getter.sample_infos)
        self.cameras: Dict[str, Cameras] = load_cameras(
            sample_config,
            self.info_getter.sample_infos,
            PoseDataGetter(
                info_getter,
                PoseDataSpec(
                    name="pose", camera=data_spec.camera, shift=data_spec.shift
                ),
                sample_config,
            ),
            TimestampDataGetter(
                info_getter,
                TimestampDataSpec(
                    name="timestamp", camera=data_spec.camera, shift=data_spec.shift
                ),
                sample_config,
            ),
            IntrinsicsDataGetter(
                info_getter,
                IntrinsicsDataSpec(
                    name="pose", camera=data_spec.camera, shift=data_spec.shift
                ),
                sample_config,
            ),
            sample_config.img_width,
            sample_config.img_height,
        )

    def get_data(self, args: SampleArgument) -> Cameras:
        return self.cameras[args.sample_info.scene][int(args.sample_info.sample)]


class RayDataGetter(CameraDataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: RayDataSpec,
        sample_config: "SampleConfig",
    ):
        super().__init__(info_getter, data_spec, sample_config)
        self.ray_generators = {
            scene: RayGenerator(cameras) for scene, cameras in self.cameras.items()
        }

    def get_data(self, args: SampleArgument) -> Tensor:
        if args.crop_size is None or args.crop_top_left is None:
            raise ValueError(
                f"RayDataGetter requires crop_size and crop_top_left to be set."
            )

        ray_generator = self.ray_generators[args.sample_info.scene]

        cam_idxs = torch.tensor(
            [self.cam_to_idx[args.sample_info.scene][args.sample_info.sample]]
        )

        ray_idxs = crop_to_ray_idxs(cam_idxs, args.crop_top_left, args.crop_size)
        rays = ray_generator.forward(ray_idxs)

        ray = torch.concat([rays.origins, rays.directions], dim=-1)
        ray = ray.reshape(args.crop_size[0], args.crop_size[1], 6).permute(2, 0, 1)

        return ray


_INFO_GETTER_BUILDERS = cast(
    Dict[str, Callable[[Path, Dict], InfoGetter]],
    {
        "pandaset": PandasetInfoGetter,
    },
)


RGB_SPEC_TYPES = {RgbDataSpec, NerfOutputSpec}


def is_data_spec_type_rgb(data_type: Type[DataSpec]) -> bool:
    return data_type in RGB_SPEC_TYPES


def get_info_getter(dataset_name: str, dataset_path: Path, data_tree: DatasetTree):
    return _INFO_GETTER_BUILDERS[dataset_name](dataset_path, data_tree)


def construct_data_getters(
    info_getter: InfoGetter,
    data_specs: Dict[str, DataSpec],
    sample_config: "SampleConfig",
):
    return {
        spec_name: spec.create_getter(info_getter, sample_config)
        for spec_name, spec in data_specs.items()
    }


class SampleConfig(BaseModel):
    rgb_flip_prob: float = 0
    random_crop: bool = False
    crop_size: Optional[Tuple[int, int]] = (512, 512)
    img_height: int = 1080
    img_width: int = 1920


class DatasetConfig(BaseModel):
    dataset_name: str = "pandaset"
    dataset_path: Path = Path("data/pandaset")
    data_specs: Dict[str, DataSpec] = {
        "rgb": RgbDataSpec(name="rgb", camera="front", shift="0m"),
        "input_ids": PromptDataSpec(name="input_ids", prompt=""),
    }
    data_tree: DatasetTree
    sample_config: SampleConfig


def construct_sample_argument(
    sample_config: SampleConfig, sample_info: SampleInfo
) -> SampleArgument:

    if sample_config.crop_size:
        crop_size = sample_config.crop_size

        if sample_config.random_crop:
            cy = int(
                torch.randint(0, sample_config.img_height - crop_size[0], (1,)).item()
            )
            cx = int(
                torch.randint(0, sample_config.img_width - crop_size[1], (1,)).item()
            )
            top_left = (cy, cx)

        else:  # center_crop
            top_left = (
                (sample_config.img_height - crop_size[0]) // 2,
                (sample_config.img_width - crop_size[1]) // 2,
            )

    else:
        top_left = crop_size = None

    do_flip = torch.rand(1).item() < sample_config.rgb_flip_prob

    return SampleArgument(
        sample_info=sample_info,
        crop_size=crop_size,
        crop_top_left=top_left,
        do_flip=do_flip,
    )


class DynamicDataset(Dataset):  # Dataset / Scene / Sample
    def __init__(
        self,
        dataset_config: DatasetConfig,
        info_getter: Optional[InfoGetter] = None,
        data_getters: Optional[Dict[str, DataGetter]] = None,
    ):
        self.dataset_config = dataset_config
        self.info_getter = info_getter or get_info_getter(
            dataset_config.dataset_name,
            dataset_config.dataset_path,
            dataset_config.data_tree,
        )
        self.data_getters = data_getters or (
            construct_data_getters(
                self.info_getter,
                dataset_config.data_specs,
                dataset_config.sample_config,
            )
        )

    @property
    def name(self) -> str:
        return self.info_getter.dataset_name

    def __len__(self) -> int:
        return len(self.info_getter.sample_infos)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.info_getter.sample_infos[idx]
        sample_argument = construct_sample_argument(
            self.dataset_config.sample_config, info
        )

        sample = {"meta": info, "args": sample_argument}

        for data_type, getter in self.data_getters.items():
            data = getter.get_data(sample_argument)
            sample[data_type] = data

        return sample
