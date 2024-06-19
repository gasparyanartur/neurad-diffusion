from typing import Optional, List, Any, Tuple, Union
from collections.abc import Iterable
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def get_device():
    # TODO: Make compatible with multi-gpu if possible

    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def show_img(img: Tensor, save_path: Optional[Path] = None):
    if isinstance(img, (list, tuple, dict, set)):
        img = list(img)
        for i in range(len(img)):
            img[i] = img[i].squeeze().to(img[0].device)

        img = torch.stack(img)

    img = img.detach().cpu()
    if img.dtype in {torch.float16, torch.bfloat16}:
        img = img.to(torch.float32)

    batch_size = len(img.shape)
    if batch_size == 4:
        fig, axes = plt.subplots(img.size(0), 1, figsize=(16, 4 * batch_size))

    elif batch_size == 3:
        fig = plt.figure()
        ax = plt.gca()

        axes = [ax]
        img = img[None, ...]

    else:
        assert False

    if not isinstance(axes, Iterable):
        axes = [axes]

    if img.shape[-1] != 3:
        img = img.permute(0, 2, 3, 1)

    for ax, img in zip(axes, img):
        ax.imshow(img)
        ax.axis("off")

    if save_path:
        fig.savefig(str(save_path))

    plt.show()


def batch_if_not_iterable(
    item: Union[Tensor, np.ndarray, Iterable], single_dim: int = 3
) -> Iterable[Any]:
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


def combine_kwargs(kwargs, extra_kwargs):
    extra_kwargs = extra_kwargs or {}
    kwargs = dict(kwargs, **extra_kwargs)
    return kwargs


from typing import Any


def get_parameter_combinations(parameters):
    names = []
    values = []
    idxs = []

    for name, value in parameters.items():
        names.append(name)
        values.append(tuple(value))
        idxs.append(0)

    idxs = tuple(idxs)

    return recurse_param_combinations(names, values, idxs, set())


def recurse_param_combinations(
    names: List[str], values: List[Tuple[Any]], idxs: Tuple[int], memory
):
    if idxs in memory:
        return []

    memory.add(idxs)

    combs = [{names[i]: values[i][idxs[i]] for i in range(len(idxs))}]

    for i in range(len(idxs)):
        new_idxs = list(idxs)
        new_idxs[i] += 1
        new_idxs = tuple(new_idxs)

        if new_idxs[i] >= len(values[i]):
            continue

        combs.extend(recurse_param_combinations(names, values, new_idxs, memory))

    return combs


def set_env(key: str, val: Any) -> None:
    os.environ[key] = str(val)


def get_env(key: str, default_val: Any = None) -> Union[str, Path]:
    val = os.environ.get(key, default_val)

    if (
        val
        and isinstance(val, str)
        and (val[0] == "/")
        and (val_path := Path(val)).exists()
    ):
        return val_path

    return val


def set_if_no_key(config, key, val):
    if not config.get(key):
        config[key] = val

    return val


def nearest_multiple(x: Union[float, int], m: int) -> int:
    return int(int(x / m) * m)
