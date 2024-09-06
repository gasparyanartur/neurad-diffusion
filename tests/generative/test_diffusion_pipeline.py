import math

import torch
from nerfstudio.pipelines.diffusion_nerf_pipeline import (
    PoseConfig,
    get_diffusion_strength,
    get_pose_augmentation,
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
