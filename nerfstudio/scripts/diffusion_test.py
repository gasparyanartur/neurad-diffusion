import torch
from torch import nn
from pathlib import Path
from nerfstudio.generative.diffusion_model import (
    StableDiffusionModel,
    read_yaml,
    read_image,
)
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as tvtf
import torchvision as tv

import argparse
import json
import logging


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser("diffusion-test")
    parser.add_argument("input_img_path", type=Path)
    parser.add_argument("output_img_path", type=Path)
    parser.add_argument(
        "--config_path", "-c", type=Path, default="configs/diffusion_model_configs.yml"
    )
    parser.add_argument(
        "--crop_size",
        help="crop size of images (before resize) [int, int]",
        nargs=2,
        default=(1024, 1024),
    )
    parser.add_argument(
        "--resize_size",
        help="downsampling size of input images (after crop) (int, int)",
        nargs=2,
        default=(512, 512),
    )

    args = parser.parse_args()
    diff_config = read_yaml(args.config_path)

    pipe = StableDiffusionModel(diff_config)
    img_transform = tvtf.Compose(
        (
            tvtf.ConvertImageDtype(torch.float32),
            tvtf.CenterCrop(args.crop_size),
            tvtf.Resize(args.resize_size, antialias=True),
        )
    )
    input_img = read_image(args.input_img_path, tf_pipeline=img_transform)
    input = {"rgb": input_img[None, ...].to(pipe.device)}
    output = pipe.get_diffusion_output(input)
    metrics = pipe.get_diffusion_metrics(output, input)
    output_img = output["rgb"][0].detach().cpu().float()

    logging.info(
        f"Metrics: \n" + "\n".join([f"\t{k} - {v}" for k, v in metrics.items()])
    )
    tv.utils.save_image(output_img, args.output_img_path)
