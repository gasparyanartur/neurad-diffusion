#!/bin/bash

#SBATCH --job-name=train_diffusionnerf
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 100G
#SBATCH --output logs/%x/slurm/%j.out
#SBATCH --array=0
#SBATCH --time 2-00:00:00
#SBATCH --account=berzelius-2024-347

source .env
if [ -z ${WANDB_API_KEY} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

export WANDB_RUN_GROUP=$name
export WANDB_ENTITY=arturruiqi
export WANDB_PROJECT=diffusionnerf

name=${NAME:-"train_diffusionnerf"}
method=${METHOD:-"diffusion-nerf"}
dataset=${DATASET:-pandaset}

job_id=${SLURM_JOB_ID:-"000000"}
task_id=${SLURM_ARRAY_TASK_ID:-"1"}
image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}



#BIND_CMD="--bind /staging:/staging --bind /workspaces:/workspaces --bind /datasets:/datasets"
BIND_CMD="--bind /proj:/proj --bind /home:/home"

execute="singularity exec \
            --nv \
            $BIND_CMD \
            --home /nerfstudio \
            --bind $PWD:/nerfstudio \
            --env PYTHONPATH=/nerfstudio \
            --env WANDB_API_KEY=$WANDB_API_KEY \
            $image_path"


if [[ -z ${ARRAY_PARAM_PATH} ]]; then
    array_params=""
else
    array_params_count=$(
        $execute python3.10 nerfstudio/scripts/param_parser.py $ARRAY_PARAM_PATH -s
    )
    if [[ $array_param_count -gt $SLURM_ARRAY_TASK_MAX ]]; then
        echo "Array parameter count $array_param_count is greater than SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_MAX - exiting"
        exit 1
    fi

    array_params=$(
        $execute python3.10 nerfstudio/scripts/param_parser.py $ARRAY_PARAM_PATH -i $task_id
    )
    if [[ -z ${array_params} ]]; then 
        echo "No array parameters found for task $task_id - exiting"
        exit 1
    fi

fi

output_dir=${OUTPUT_DIR:="outputs/$name"}
mkdir -p $output_dir

dataset_root=${DATASET_ROOT:-data/$dataset}
if [ ! -d $dataset_root ]; then
    echo "Dataset root $dataset_root does not exist. Exiting."
    exit 1
fi

echo "Starting training with job_id $job_id"
echo "Output directory: $output_dir"
echo "Dataset root: $dataset_root"
echo "Array parameters: $array_params"
echo "Array parameters data: $array_params_data"
echo "Method: $method"

experiment_name=${EXPERIMENT_NAME:-$(date +%Y%m%d_%H%M%S)_$job_id}
experiment_name=${experiment_name}

if [ -z ${NERF_CHECKPOINT_PATH} ]; then
    checkpoint_cmd=""
else
    checkpoint_cmd="--pipeline.nerf_checkpoint $NERF_CHECKPOINT_PATH"
fi

augment_strategy=${AUGMENT_STRATEGY:-"partial_linear"}


$execute python3.10 -u nerfstudio/scripts/train.py \
    $method \
    --output-dir $output_dir \
    --vis wandb \
    --experiment-name $experiment_name \
    $checkpoint_cmd \
    --pipeline.diffusion_model.dtype "fp32" \
    --pipeline.augment_strategy $augment_strategy \
    $@ \
    $array_params 

chmod 775 -R $output_dir

#
#EOF