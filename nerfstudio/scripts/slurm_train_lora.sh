#!/bin/bash

#SBATCH --job-name=train_lora
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 100G
#SBATCH --output logs/%x/slurm/%j.out
#SBATCH --array=0

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

# SLURM_JOB_ID is unque for each job
# SLURM_ARRAY_JOB_ID is the same for each job, with different SLURM_ARRAY_TASK_ID
# We want to use the unique job id in the actual python script, but use params corresponding to task id

job_id=${SLURM_JOB_ID:-"000000"}
task_id=${SLURM_ARRAY_TASK_ID:-"0"}

execute="singularity exec \
            --nv \
            --bind $PWD:/nerfstudio \
            --bind /staging:/staging \
            --bind /workspaces:/workspaces \
            --bind /datasets:/datasets \
            --env PYTHONPATH=/nerfstudio \
            --env WANDB_API_KEY=$wandb_api_key \
            --env LORA_TRAIN_CONFIG=$lora_train_config \
            --home /nerfstudio \
            $image_path"

array_param_path=${ARRAY_PARAM_PATH:-nerfstudio/scripts/params/train_lora_rank_lr.json}

array_params_count=$($execute python3.10 nerfstudio/scripts/param_parser.py $array_param_path -s)
if [[ $array_param_count -gt $SLURM_ARRAY_TASK_MAX ]]; then
    echo "Array parameter count $array_param_count is greater than SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_MAX - exiting"
    exit 1
fi

array_params=$($execute python3.10 nerfstudio/scripts/param_parser.py $array_param_path -i $task_id)
if [[ -z ${array_params} ]]; then 
    echo "No array parameters found for task $task_id - exiting"
    exit 1
fi

$execute accelerate launch \
        --num_processes=$num_processes \
        --num_machines=$num_machines \
        --dynamo_backend=$dynamo_backend \
        --mixed_precision=$mixed_precision \
        --main_process_port=$main_process_port \
    nerfstudio/scripts/train_lora.py \
        --job_id $job_id \
        $array_params \
        ${@:1} 