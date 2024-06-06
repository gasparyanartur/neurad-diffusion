#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/finetune/slurm/%j.out
#SBATCH --partition=zprod 
#SBATCH --job-name=finetune

main_process_port=${MAIN_PROCESS_PORT:-29500}
num_processes=${NUM_PROCESSES:-1}
num_machines=${NUM_MACHINES:-1}
dynamo_backend=${DYNAMO_BACKEND:-"no"}
mixed_precision=${MIXED_PRECISION:-"no"}

wandb_api_key=${WANDB_API_KEY}
if [ -z ${wandb_api_key} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

#image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/neuraddiffusion-24_05_24.sif"}
config_path=${CONFIG_PATH:-"configs/hal-configs/train_model.yml"}

#singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --num_processes=$num_processes --num_machines=$num_machines --dynamo_backend=$dynamo_backend --mixed_precision=$mixed_precision --main_process_port=$main_process_port scripts/run_train_lora.py $config_path
singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /staging:/staging \
    --bind /workspaces:/workspaces \
    --bind /datasets:/datasets \
    --env PYTHONPATH=/nerfstudio \
    --env WANDB_API_KEY=$wandb_api_key \
    --home /nerfstudio \
    $image_path \
    accelerate launch \
        --num_processes=$num_processes \
        --num_machines=$num_machines \
        --dynamo_backend=$dynamo_backend \
        --mixed_precision=$mixed_precision \
        --main_process_port=$main_process_port \
    nerfstudio/scripts/run_train_lora.py \
        $config_path