#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/train_lora/slurm/%j.out
#SBATCH --partition=zprodlow
#SBATCH --job-name=finetune

if [ -z ${WANDB_API_KEY} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

lora_train_config=${LORA_TRAIN_CONFIG:-"configs/hal-configs/train_lora.yml"}
image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}

main_process_port=${MAIN_PROCESS_PORT:-29500}
num_processes=${NUM_PROCESSES:-1}
num_machines=${NUM_MACHINES:-1}
dynamo_backend=${DYNAMO_BACKEND:-"no"}
mixed_precision=${MIXED_PRECISION:-"no"}

#singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --num_processes=$num_processes --num_machines=$num_machines --dynamo_backend=$dynamo_backend --mixed_precision=$mixed_precision --main_process_port=$main_process_port scripts/run_train_lora.py $config_path
singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /staging:/staging  \
    --bind /workspaces:/workspaces \
    --bind /datasets:/datasets \
    --env PYTHONPATH=/nerfstudio \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env LORA_TRAIN_CONFIG=$lora_train_config \
    --home /nerfstudio \
    $image_path \
    accelerate launch \
        --num_processes=$num_processes \
        --num_machines=$num_machines \
        --dynamo_backend=$dynamo_backend \
        --mixed_precision=$mixed_precision \
        --main_process_port=$main_process_port \
    nerfstudio/scripts/run_train_lora.py \
        ${@:1} 