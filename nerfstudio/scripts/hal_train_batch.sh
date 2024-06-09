diffusion_config_path="/staging/agp/masterthesis/nerf-thesis-shared/output/finetune/sd/335491/final_weights/config.yml"
neurad_path="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331982/imaginedriving-001-331982-1/imaginedriving/2024-05-25_130744"

neurad_checkpoint_path="${neurad_path}/nerfstudio_models/step-000020000.ckpt"
augment_strategy="partial_const"
augment_phase_step=0

#WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' sbatch nerfstudio/scripts/halnew.sh imaginedriving --pipeline.augment_strategy=$augment_strategy --pipeline.nerf_checkpoint=$neurad_checkpoint_path
WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' srun --gpus 1 --nodes 1 -c 32 --job-name neurad_imaginedriving --partition zprod bash nerfstudio/scripts/halnew.sh imaginedriving --pipeline.augment_strategy $augment_strategy --pipeline.nerf_checkpoint $neurad_checkpoint_path