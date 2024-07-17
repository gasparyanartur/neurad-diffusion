import typing
from pydantic import BaseModel
from typing import Any, Dict, List, Set, Union, Optional, Tuple, cast
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


def setup_project(config_path: Path, override_values: Dict[str, Any] = {}):
    logging.getLogger().setLevel(logging.INFO)

    if not torch.cuda.is_available():
        logging.warning(
            f"CUDA not detected. Running on CPU. The code is not supported for CPU and will most likely give incorrect results. Proceed with caution."
        )

    if "config_path" in override_values:
        config_path = Path(override_values.pop("config_path"))

    if config_path is None:
        project_dir = get_env("PROJECT_DIR") or Path.cwd()
        config_path = project_dir / "proj_config.yml"
        if not config_path.exists():
            raise ValueError(f"No config path specified")

    else:
        if not config_path.exists():
            raise ValueError(f"Could not find config at specified path: {config_path}")

    config = read_yaml(config_path)
    config.update(override_values)

    project_dir = config.get("project_path") or get_env("PROJECT_DIR", Path.cwd())
    cache_dir = config.get("cache_dir") or get_env("CACHE_DIR", project_dir / ".cache")

    set_env("HF_HUB_CACHE", cache_dir / "hf")  # Huggingface cache dir
    set_env("MPLCONFIGDIR", cache_dir / "mpl")  # Matplotlib cache dir

    return config


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


def read_data_tree(
    data_tree: Union[Dict, Path], scene_name_len: int = 3, sample_name_len: int = 2
):
    if isinstance(data_tree, (str, Path)):
        data_tree = Path(data_tree)
        data_tree = read_yaml(data_tree)

    dataset_dict = {}
    for dataset_name, dataset in data_tree.items():
        scene_dict = {}
        for scene_name, scene in dataset.items():
            if isinstance(scene, str):
                if scene == "*":
                    sample_list = (None, None, None)

                elif RANGE_SEP in scene:
                    sample_list = scene.strip().split(RANGE_SEP)
                    assert len(sample_list) == 3

                    start_range = int(sample_list[0]) if sample_list[0] != "" else None
                    end_range = int(sample_list[1]) if sample_list[1] != "" else None
                    skip_range = int(sample_list[2]) if sample_list[2] != "" else None

                    sample_list = (start_range, end_range, skip_range)

                else:
                    raise NotImplementedError

            else:
                sample_list = []
                for sample in scene:
                    sample = str(sample)
                    sample = sample.replace(" ", "")
                    if sample.isdigit():
                        sample = sample.rjust(sample_name_len, "0")
                        sample_list.append(sample)
                    else:
                        assert "-" in sample
                        sample_from, sample_to = sample.split("-")
                        assert sample_from.isdigit() and sample_to.isdigit()
                        sample_list.extend(
                            iter_numeric_names(
                                sample_from, sample_to, fixed_len=sample_name_len
                            )
                        )

            scene_name = str(scene_name)
            if scene_name.isdigit():
                scene_name = scene_name.rjust(scene_name_len, "0")
                scene_dict[scene_name] = sample_list
            else:
                assert "-" in scene_name
                scene_from, scene_to = scene_name.split("-")
                assert scene_from.isdigit() and scene_to.isdigit()

                for new_scene_name in iter_numeric_names(
                    scene_from, scene_to, fixed_len=scene_name_len
                ):
                    scene_dict[new_scene_name] = sample_list

        dataset_dict[dataset_name] = scene_dict
    return dataset_dict


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


class DataSpec(BaseModel):
    name: str


class CameraDataSpec(DataSpec):
    camera: str
    shift: str


class LidarDataSpec(DataSpec):
    shift: str


class RgbDataSpec(CameraDataSpec):
    num_channels: int = 3
    width: int = 1920
    height: int = 1080
    dtype: str = "fp32"
    normalize: bool = True


class NerfOutputSpec(RgbDataSpec):
    nerf_output_path: str
    data_name: str
    final_size: int = 512
    crop_to_final_ratio: float = 2.0


class PromptDataSpec(DataSpec):
    prompt: str


class PoseDataSpec(CameraDataSpec): ...


class IntrinsicsDataSpec(CameraDataSpec): ...


class TimestampDataSpec(CameraDataSpec): ...


class RayDataSpec(CameraDataSpec): ...


_SPLIT_SCORE = {"train": 0, "validation": 1, "test": 2}


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

def get_cam_idxs(infos: Iterable[SampleInfo]) -> Dict[str, Dict[str, int]]:
    cam_to_idx = {}
    for info in infos:
        if info.scene not in cam_to_idx:
            cam_to_idx[info.scene] = {}

        cam_to_idx[info.scene][info.sample] = len(cam_to_idx[info.scene])
    return cam_to_idx


@dataclass
class SampleArguments:
    dataset_path: Path,
    sample_info: SampleInfo,
    crop_size: Optional[Tuple[int, int]] = None
    crop_top_left: Optional[Tuple[int, int]] = None
    do_flip: Optional[bool] = None



class InfoGetter(ABC):
    def __init__(self, dataset_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    @abstractmethod
    def get_scenes(self, dataset_path: Path, dataset: str, split: str):
        raise NotImplementedError

    @abstractmethod
    def get_samples(
        self,
        dataset_path: Path,
        dataset: str,
        scene: str,
        split: str,
        spec: Optional[DataSpec] = None,
    ) -> List[SampleInfo]:
        raise NotImplementedError

    @abstractmethod
    def get_path(self, dataset_path: Path, info: SampleInfo, spec: DataSpec):
        raise NotImplementedError

    @abstractmethod
    def get_suffix(self, spec: DataSpec) -> str:
        raise NotImplementedError

    def parse_tree(
        self, dataset_path: Path, data_tree: Dict[str, Any]
    ) -> List["SampleInfo"]:

        sample_infos = []

        for dataset, dataset_dict in data_tree.items():
            for split, split_dict in dataset_dict.items():
                existing_scenes = self.get_scenes(dataset_path, dataset, split)

                for scene, sample_list in split_dict.items():
                    if scene not in existing_scenes:
                        continue

                    if isinstance(sample_list, tuple) and len(sample_list) == 3:
                        start_range, end_range, skip_range = sample_list

                        samples = sorted(
                            self.get_samples(dataset_path, dataset, scene, split)
                        )

                        start_range = start_range or 0
                        end_range = end_range or len(samples)
                        skip_range = skip_range or 1

                        sample_infos.extend(samples[start_range:end_range:skip_range])

                    else:
                        if not sample_list:
                            continue

                        assert isinstance(sample_list, list)
                        assert isinstance(sample_list[0], str)

                        sample_infos.extend(
                            [
                                SampleInfo(dataset, scene, sample, split)
                                for sample in sample_list
                            ]
                        )

        return sample_infos


class PandasetInfoGetter(InfoGetter):
    def __init__(self) -> None:
        super().__init__("pandaset")

    def get_scenes(self, dataset_path: Path, dataset: str, split: str) -> Set[str]:
        return set(path.stem for path in dataset_path.iterdir())

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
        dataset_path: Path,
        dataset: str,
        scene: str,
        split: str,
        spec: Optional[DataSpec] = None,
    ) -> List[str]:
        if isinstance(spec, LidarDataSpec):
            sample_dir = dataset_path / scene / "lidar"
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
            sample_dir = dataset_path / scene / "camera" / spec.camera
            suffix = self.get_suffix(spec)

        else:
            sample_dir = dataset_path / scene / "camera" / "front"
            suffix = ""

        return [path.stem for path in sample_dir.glob(f"*{suffix}")]

    def get_path(self, dataset_path: Path, info: SampleInfo, spec: DataSpec) -> Path:
        if isinstance(spec, LidarDataSpec):
            sample_path = dataset_path / info.scene / "lidar" / info.sample

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
                dataset_path / info.scene / "camera" / spec.camera / info.sample
            )

        elif isinstance(spec, PoseDataSpec):
            sample_path = dataset_path / info.scene / "camera" / spec.camera / "poses"

        elif isinstance(spec, IntrinsicsDataSpec):
            sample_path = (
                dataset_path / info.scene / "camera" / spec.camera / "intrinsics"
            )

        elif isinstance(spec, CameraDataSpec):
            sample_path = (
                dataset_path / info.scene / "camera" / spec.camera / info.sample
            )

        else:
            sample_path = dataset_path / info.scene / "camera" / "front" / info.sample

        sample_path = sample_path.with_suffix(self.get_suffix(spec))
        return sample_path


class DataGetter(ABC):
    def __init__(self, info_getter: InfoGetter, data_spec: DataSpec) -> None:
        super().__init__()
        self.info_getter = info_getter
        self.data_spec = data_spec

    def get_data_path(self, dataset_path: Path, info: SampleInfo) -> Path:
        return self.info_getter.get_path(dataset_path, info, self.data_spec)

    @abstractmethod
    def get_data(
        self, args: SampleArguments
    ) -> Any:
        raise NotImplementedError


class PromptDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: PromptDataSpec,
    ):
        super().__init__(info_getter, data_spec)
        self.data_spec = data_spec

    def get_data(
        self, args: SampleArguments
    ) -> str:
        return self.data_spec.prompt


class RGBDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: RgbDataSpec,
        transform: Optional[tvtf.Compose] = None,
    ):
        super().__init__(info_getter, data_spec)
        self.data_spec = data_spec

        if transform is None:
            transform = tvtf.Compose(
                [
                    (
                        tvtf.ConvertImageDtype(DTYPE_CONVERSION[data_spec.dtype])
                        if data_spec.normalize
                        else tvtf.ToDtype(DTYPE_CONVERSION[data_spec.dtype])
                    ),
                    tvtf.Resize((data_spec.height, data_spec.width), antialias=True),
                ]
            )

        self.transform = transform
        self.extra_transform: Optional[tvtf.Compose] = None

    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
        rgb_path = self.get_data_path(args.dataset_path, args.sample_info)
        rgb = read_image(rgb_path, self.transform)

        if (args.crop_top_left and not args.crop_size) or (
            args.crop_size and not args.crop_top_left
        ):
            raise ValueError(
                f"Both crop_top_left and crop_size must be set if either is set."
            )

        if args.crop_top_left and args.crop_size:
            rgb = tvtf.functional.crop(
                rgb, *args.crop_top_left, *args.crop_size
            )

        rgb = tvtf.functional.hflip(rgb) if args.do_flip else rgb

        return rgb


class LidarDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: LidarDataSpec,
    ):
        super().__init__(info_getter, data_spec)
        self.data_spec = data_spec

    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
        path = self.get_data_path(args.dataset_path, args.sample_info)
        data = pd.read_pickle(path)

        return data


class PoseDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: PoseDataSpec,
    ):
        super().__init__(info_getter, data_spec)

    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
        file_path = self.get_data_path(args.dataset_path, args.sample_info)
        poses = load_json(file_path)

        pose = poses[int(args.sample_info.sample)]
        pose = _pandaset_pose_to_matrix(pose)

        return pose.to(dtype=torch.float32, device=pose.device)


class IntrinsicsDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: IntrinsicsDataSpec,
    ):
        super().__init__(info_getter, data_spec)

    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
        file_path = self.get_data_path(args.dataset_path, args.sample_info)

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
    ):
        super().__init__(info_getter, data_spec)

    def get_data(
        self, args: SampleArguments
    ) -> float:
        file_path = self.get_data_path(args.dataset_path, args.sample_info)

        all_timestamps = load_json(file_path)
        timestamp = all_timestamps[int(args.sample_info.sample)]
        return timestamp


class NerfOutputDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: NerfOutputSpec,
    ):
        super().__init__(info_getter, data_spec)

        self.transform = tvtf.Compose(
            [
                (
                    tvtf.ConvertImageDtype(DTYPE_CONVERSION[data_spec.dtype])
                    if data_spec.normalize
                    else tvtf.ToDtype(DTYPE_CONVERSION[data_spec.dtype])
                ),
                tvtf.CenterCrop(
                    int(data_spec.crop_to_final_ratio * data_spec.final_size)
                ),
                tvtf.Resize(data_spec.final_size, antialias=True),
            ]
        )

    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
        path = self.get_data_path(args.dataset_path, args.sample_info)
        data = read_image(path, self.transform)

        return data


class CameraDataGetter(DataGetter):
    def __init__(self, info_getter, data_spec: CameraDataSpec):
        super().__init__(info_getter, data_spec)

        self.is_loaded = False
        self.cameras: Dict[str, Cameras] = {}

    def load_cameras(
        self,
        dataset_path: Path,
        infos: List[SampleInfo],
        pose_getter: PoseDataGetter,
        timestamp_getter: TimestampDataGetter,
        intrinsics_getter: IntrinsicsDataGetter,
        img_width: int = 1920,
        img_height: int = 1080,
    ):
        # dataset / scene / sample
        self.cameras = {}

        poses = {}
        timestamps = {}
        intrinsics = {}
        sensor_idxs = {}

        # Assume const dataset
        dataset = infos[0].dataset

        unique_scenes = {info.scene for info in infos}
        data_tree = {
            scene: [
                info
                for info in infos
                if info.scene == scene and info.dataset == dataset
            ]
            for scene in unique_scenes
        }

        for scene, samples in data_tree.items():
            poses = torch.stack(
                [pose_getter.get_data(SampleArguments(dataset_path, info)) for info in samples]
            )
            timestamps = torch.tensor(
                [timestamp_getter.get_data(SampleArguments(dataset_path, info)) for info in samples]
            )
            intrinsics = torch.stack(
                [intrinsics_getter.get_data(SampleArguments(dataset_path, info)) for info in samples]
            )
            sensor_idxs = torch.tensor([int(info.sample) for info in samples])

            self.cameras[scene] = create_cameras_for_sequence(
                poses, timestamps, intrinsics, sensor_idxs, img_width, img_height
            )

        self.is_loaded = True

    def get_data(
        self, args: SampleArguments
    ) -> Cameras:
        if not self.is_loaded:
            raise ValueError(
                f"Tried to call CameraDataGetter without loading data first."
            )

        return self.cameras[args.sample_info.scene][int(args.sample_info.sample)]


class RayDataGetter(CameraDataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: RayDataSpec,
    ):
        super().__init__(info_getter, data_spec)
        self.ray_generators = {}

    def load_cameras(
        self,
        dataset_path: Path,
        infos: List[SampleInfo],
        pose_getter: PoseDataGetter,
        timestamp_getter: TimestampDataGetter,
        intrinsics_getter: IntrinsicsDataGetter,
        img_width: int = 1920,
        img_height: int = 1080,
    ):
        super().load_cameras(
            dataset_path,
            infos,
            pose_getter,
            timestamp_getter,
            intrinsics_getter,
            img_width,
            img_height,
        )

        self.ray_generators = {
            scene: RayGenerator(cameras) for scene, cameras in self.cameras.items()
        }
        self.cam_to_idx = get_cam_idxs(infos)


    def get_data(
        self, args: SampleArguments
    ) -> Tensor:
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
        ray = ray.reshape(
            args.crop_size[0], args.crop_size[1], 6
        ).permute(2, 0, 1)

        return ray

INFO_GETTER_BUILDERS: Dict[str, Callable[[], InfoGetter]] = {
    "pandaset": PandasetInfoGetter,
}

DATA_GETTER_BUILDERS: Dict[str, Callable[[InfoGetter, DataSpec], DataGetter]] = cast(
    Dict[str, Callable[[InfoGetter, DataSpec], DataGetter]],
    {
        "ref-rgb": NerfOutputDataGetter,
        "gt-rgb": RGBDataGetter,
        "gt": RGBDataGetter,
        "rgb": RGBDataGetter,
        "lidar": LidarDataGetter,
        "prompt": PromptDataGetter,
        "intrinsics": IntrinsicsDataGetter,
        "pose": PoseDataGetter,
    },
)


@dataclass(init=True, slots=True, frozen=True)
class ConditioningSignalInfo:
    cn_type: str
    num_channels: int
    camera: str
    data_type: Optional[str] = None

    @staticmethod
    def from_signal_name(name: str) -> Optional["ConditioningSignalInfo"]:
        pattern_match = CN_SIGNAL_PATTERN.match(name)
        if not pattern_match:
            return None

        group = pattern_match.groupdict()
        group["num_channels"] = int(group["num_channels"])
        if group["cn_type"] in DATA_GETTER_BUILDERS:
            group["data_type"] = group["cn_type"]

        return ConditioningSignalInfo(**group)

    @property
    def name(self):
        return f"cn_{self.cn_type}_{self.num_channels}_{self.camera}"


class DynamicDataset(Dataset):  # Dataset / Scene / Sample
    def __init__(
        self,
        dataset_path: Path,
        data_tree: Dict[str, Any],
        info_getter: InfoGetter,
        data_getters: Dict[str, DataGetter],
        data_transforms: Dict[str, Callable[[str], int]] = None,
        preprocess_func=None,
    ):
        self.sample_infos: list[SampleInfo] = info_getter.parse_tree(
            dataset_path, data_tree
        )
        self.info_getter = info_getter
        self.data_getters = data_getters
        self.dataset_path = dataset_path
        self.preprocess_func = preprocess_func

        for getter in data_getters.values():
            if getter.data_spec["data_type"] == "camera":
                cam_getter = typing.cast(CameraDataGetter, getter)
                cam = cam_getter.data_spec.get("camera", "front")
                cam_getter.load_cameras(
                    dataset_path,
                    self.sample_infos,
                    PoseDataGetter(info_getter, {"data_type": "pose", "camera": cam}),
                    TimestampDataGetter(
                        info_getter, {"data_type": "timestamp", "camera": cam}
                    ),
                    IntrinsicsDataGetter(
                        info_getter, {"data_type": "intrinsics", "camera": cam}
                    ),
                    img_width=1920,  # Hardcoded for now
                    img_height=1080 if cam != "back" else 1080 - 250,
                )

        self.data_transforms = {**data_transforms} if data_transforms else {}
        for data_type in data_getters.keys():
            if data_type not in self.data_transforms:
                self.data_transforms[data_type] = []

        self.reset_index()

    def reset_index(self):
        self.idxs = np.arange(len(self.sample_infos))

    def shuffle_index(self):
        np.random.shuffle(self.idxs)

    def limit_size(self, size: float | int):
        if size is None:
            size = self.true_len

        elif isinstance(size, float):
            size = int(self.true_len * size)

        self.idxs = self.idxs[:size]

    @property
    def true_len(self):
        return len(self.sample_infos)

    @classmethod
    def from_config(cls, dataset_config: Union[Path, Dict[str, Any]], **kwargs):
        if isinstance(dataset_config, Path):
            dataset_config = read_yaml(dataset_config)
            dataset_config = typing.cast(Dict[str, Any], dataset_config)

        dataset_path = Path(dataset_config["path"])
        data_tree = read_data_tree(dataset_config["data_tree"])
        dataset_name = dataset_config["dataset"]

        info_getter_factory = INFO_GETTER_BUILDERS[dataset_name]
        info_getter = info_getter_factory()

        data_getters = {}

        for spec_name, spec in dataset_config["data_getters"].items():
            data_type = spec.get("data_type")
            if not data_type:
                if spec_name in DATA_GETTER_BUILDERS:
                    data_type = spec_name

                elif signal_info := ConditioningSignalInfo.from_signal_name(spec_name):
                    if not signal_info.data_type:
                        continue

                    data_type = signal_info.data_type

                else:
                    raise NotImplementedError(
                        f"Could not find a builder for {data_type}"
                    )

            spec["name"] = spec_name

            data_getter_factory = DATA_GETTER_BUILDERS[data_type]
            data_getter = data_getter_factory(info_getter, spec)

            data_getters[spec_name] = data_getter

        return DynamicDataset(
            dataset_path, data_tree, info_getter, data_getters, **kwargs
        )

    @property
    def name(self) -> str:
        return self.info_getter.dataset_name

    def iter_range(
        self, id: int = 0, id_start: int = 0, id_stop: int = 0, verbose: bool = True
    ):
        """Iterate cyclically with a range, such that a specific offset is matched with the right index.
            Example: (id: 14, id_start: 10, id_stop: 25)
                id=10 should be assigned i=0, id=11 gets i=1, etc...
                This continues until id=25, after which it repeats at id=10.
                We therefore get the map:
                {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, ...,
                 10: 15, 11: 16, 12: 17, 13: 18, 14: 19, ...}
                Thus, id=14 will be assigned indexes 4, 19, 34, 49, ..., until the dataset is exhausted.

        Args:
            id: Offset between the id range and index range. Defaults to 0.
            id_start: Starting index of the cycle. Defaults to 0.
            id_stop (int, optional): Final id in the index range before restarting the cycle. Defaults to 0.
            verbose: Whether or not to log each index. Defaults to True.

        Yields:
            Sample at specified index.
        """
        assert id_start <= id <= id_stop

        skip = id_stop - id_start + 1
        start = id - id_start
        stop = len(self)

        for i in range(start, stop, skip):
            if verbose:
                logging.info(f"Sample {i}/{stop} (skip: {skip})")

            yield self[i]

    def __len__(self) -> int:
        return len(self.idxs)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = self.idxs[idx]

        info = self.sample_infos[i]
        sample = {}

        for data_type, getter in self.data_getters.items():
            data = getter.get_data(self.dataset_path, info)

            for data_transform in self.data_transforms[data_type]:
                data = data_transform(data)

            sample[data_type] = data

        if self.preprocess_func:
            sample = self.preprocess_func(sample)

        return sample

    def iter_attrs(
        self, attrs: Iterable[str]
    ) -> Generator[Tuple[Any, ...], None, None]:
        for sample in self:
            yield tuple(sample[attr] for attr in attrs)

    def get_matching(
        self, other: "DynamicDataset", match_attrs: Iterable[str] = ("scene", "sample")
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        match_dict = {}
        for sample_self in self:
            sample_query = tuple(sample_self[attr] for attr in match_attrs)
            match_dict[sample_query] = sample_self

        matches = []
        for sample_other in other:
            sample_query = tuple(sample_other[attr] for attr in match_attrs)
            if sample_query in match_dict:
                sample_self = match_dict[sample_query]
                matches.append((sample_self, sample_other))

        return matches
