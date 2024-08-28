from pathlib import Path
import typing
from typing import Optional
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
import torch
from transformers import AutoTokenizer
from transformers import CLIPTextModel


from nerfstudio.generative.diffusion_model import DiffusionModelId
from nerfstudio.generative.diffusion_model import (
    import_encoder_class_from_model_name_or_path,
)


from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from diffusers.models import ControlNetModel

from nerfstudio.generative.dynamic_dataset import make_img_tf_pipe, read_image
from nerfstudio.generative.utils import show_img

model_id = DiffusionModelId.sd_v2_1
revision: str = "main"
variant: Optional[str] = None

device = torch.device("cuda")
dtype = torch.float16


ex_img_path = Path("data/pandaset/001/camera/front_camera/00.jpg")
ex_img = read_image(
    ex_img_path, make_img_tf_pipe(dtype=dtype, crop_type="center"), device=device
)
show_img(ex_img)

base_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, revision=revision, variant=variant
)

controlnet = ControlNetModel.from_unet(base_pipe.unet)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    unet=base_pipe.unet,
    text_encoder=base_pipe.text_encoder,
    vae=base_pipe.vae,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    revision=revision,
    variant=variant,
)
pipe.to(device=device, dtype=dtype)


pipe(prompt="", image=ex_img[None, ...], control_image=ex_img[None, ...])
