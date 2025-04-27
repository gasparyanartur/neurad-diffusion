# Diffusion models for novel view synthesis in autonomous driving

This is the repository for our Master's Thesis project. The full report can be found [here](https://odr.chalmers.se/items/4089cceb-4124-4737-87b8-9d1246cb8c2b)

This repo is forked from [NeuRAD-studio](https://github.com/georghess/neurad-studio), an Autonomous Driving focused extension of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) developed by our supervisors at Zenseact. 

Our main contributions are:

* Implementing a Latent Diffusion Model based on HuggingFace, which is compatible with both ControlNet and LoRA at the same time. Found in `nerfstudio/generative/diffusion_model.py`.
* Implementing a dynamic dataset that can be configured to load various data sources, including lidar, images, prompts, etc. Found in `nerfstudio/generative/dynamic_dataset.py`
* Implementing a NeRF pipeline that shifts poses during training, then uses a given diffusion model to generate a target image to optimize towards. Found in  `nerfstudio/pipelines/diffusion_nerf_pipeline.py`
* Implementing a LoRA training script that uses our dynamic dataset and diffusion model to train an autonomous driving-focused diffusion model. Found in `nerfstudio/scripts/train_lora.py`
