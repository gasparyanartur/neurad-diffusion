#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task 8
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/neurad/slurm/%j.out
#SBATCH --partition=zprodlow 
#SBATCH --job-name=neurad


export WANDB_RUN_GROUP=$name
export WANDB_ENTITY=arturruiqi
export WANDB_PROJECT=neurad

method=${METHOD:-neurad}
dataset=${DATASET:-pandaset}

image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
config_path=${CONFIG_PATH:-"configs/hal-configs/train_model.yml"}
workdir=${WORKDIR:-"/home/s0001900/workspace/imaginedriving"}

singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --num_processes=$num_processes --num_machines=$num_machines scripts/run_train_lora.py $config_path