


WANDB_API_KEY=$WANDB_API_KEY CONFIG_PATH="configs/full-training/sd-pandaset-scene.yml" NUM_PROCESSES=8 sbatch --partition=zprodlow nerfstudio/scripts/hal_finetune.sh --lora_rank $lora_rank --noise_strength $noise_strength