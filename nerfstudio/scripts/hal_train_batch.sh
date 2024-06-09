lora_weights="/staging/agp/masterthesis/nerf-thesis-shared/output/finetune/sd/335491/final_weights/pytorch_lora_weights.safetensors"
neurad_path="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331982/imaginedriving-001-331982-1/imaginedriving/2024-05-25_130744"
neurad_checkpoint_path="${neurad_path}/nerfstudio_models/step-000020000.ckpt"

augment_strategy="partial_const"
#augment_strategy="partial_linear"
#augment_strategy="none"

max_shift=5
#max_shift=3
#max_shift=0

max_rot=3.141
#max_rot=1.571
#max_rot=0.785
#max_rot=0

#WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' python3.10 nerfstudio/scripts/train.py imaginedriving --output-dir outputs/imaginedriving --vis wandb --experiment-name "test-run" --pipeline.augment_strategy $augment_strategy --pipeline.nerf_checkpoint $neurad_checkpoint_path pandaset-data --data data/pandaset --sequence 001 --cameras front 
WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' sbatch nerfstudio/scripts/halnew.sh imaginedriving --pipeline.augment_strategy $augment_strategy --pipeline.nerf_checkpoint $neurad_checkpoint_path --pipeline.augment_phase_step 0 --pipeline.diffusion_model.lora_weights $lora_weights --pipeline.diffusion_model.dtype" "fp16" --pipeline.augment_max_strength $max_shift 0 0 0 0 $max_rot
