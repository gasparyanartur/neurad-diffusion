noise_strength=0.1
#noise_strength=0.2
#noise_strength=0.3

lora_rank=4
#lora_rank=8
#lora_rank=16
#lora_rank=32
#lora_rank=64
#lora_rank=128

WANDB_API_KEY=$WANDB_API_KEY CONFIG_PATH="configs/full-training/sd-pandaset-scene.yml" NUM_PROCESSES=8 sbatch --partition=zprodlow nerfstudio/scripts/hal_finetune.sh --lora_rank $lora_rank --noise_strength $noise_strength