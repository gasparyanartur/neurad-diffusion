import pickle
import math
from typing import cast

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.pipelines.diffusion_nerf_pipeline import (
    PoseConfig,
    get_cam_rays_from_bundle,
    get_diffusion_strength,
    get_pose_augmentation,
    is_cam_ray,
    unfold_ray_vec,
    upsample_rays,
)


def test_get_diffusion_strength():
    assert get_diffusion_strength(2000, 0.5, 2000, 5000, 10) == 0.5
    assert math.isclose(
        get_diffusion_strength(5000, 0.5, 2000, 5000, 10), 1 / 10, abs_tol=1e-3
    )

    assert get_diffusion_strength(3500, 0.5, 2000, 5000, 10) == 0.25


def test_get_pose_augmentation():
    event = torch.tensor([True, False, False, False, False, True])
    assert torch.allclose(
        get_pose_augmentation(
            0, PoseConfig(pos_x=4, rot_z=45), 1000, event, rng=torch.manual_seed(0)
        ),
        torch.zeros(6),
    )

    assert torch.allclose(
        get_pose_augmentation(
            500,
            PoseConfig(pos_x=4, rot_z=45),
            1000,
            event,
            rng=torch.manual_seed(0),
        ),
        torch.tensor([4, 0, 0, 0, 0, math.radians(45)])
        * torch.empty_like(event)
        .float()
        .uniform_(-0.5, 0.5, generator=torch.manual_seed(0)),
    )

    assert torch.allclose(
        get_pose_augmentation(
            1000,
            PoseConfig(pos_x=4, rot_z=45),
            1000,
            event,
            rng=torch.manual_seed(0),
        ),
        torch.tensor([4, 0, 0, 0, 0, math.radians(45)])
        * torch.empty_like(event)
        .float()
        .uniform_(-1, 1, generator=torch.manual_seed(0)),
    )


def test_unfold_ray_vec():
    B = 2
    H = 128
    W = 128
    C = 3

    ray_vec = torch.randn(B * H * W, C)
    assert unfold_ray_vec(ray_vec, (H, W)).shape == (B, C, H, W)


def test_get_cam_rays_from_bundle():
    B = 2
    H = 128
    W = 128

    with open(f"tests/mock_data/ray_bundle-B{B}_H{H}_W{W}.pkl", "rb") as f:
        ray_bundle = cast(RayBundle, pickle.load(f))

    is_cam = is_cam_ray(ray_bundle)
    assert ray_bundle.origins[is_cam].shape == (B * H * W, 3)

    origins, directions = get_cam_rays_from_bundle(ray_bundle, (H, W))
    assert origins.shape == (B, 3, H, W)
    assert directions.shape == (B, 3, H, W)


def test_upsample_rays():
    B = 1
    H = 10
    W = 10
    F = 2

    rays = torch.zeros((B, 3, H, W))
    assert upsample_rays(rays, F).shape == (B, 3, H * F, W * F)
