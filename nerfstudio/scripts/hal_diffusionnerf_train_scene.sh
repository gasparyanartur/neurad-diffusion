#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/train_diffusionnerf/slurm/%j.out
#SBATCH --array=0-15
#SBATCH --job-name=train_diffusionnerf
#SBATCH --partition zprodlow

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
data_task_id=0      # TODO: Some sort of index
image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}

array_param_path=${ARRAY_PARAM_PATH:-nerfstudio/scripts/params/${name}_method.json}
array_param_data_path=${ARRAY_PARAM_DATA_PATH:-nerfstudio/scripts/params/${name}_data.json}

execute="singularity exec \
            --nv \
            --bind $PWD:/nerfstudio \
            --bind /staging:/staging \
            --bind /workspaces:/workspaces \
            --bind /datasets:/datasets \
            --env WANDB_API_KEY=$WANDB_API_KEY \
            $image_path"

array_params_count=$(
    $execute python3.10 nerfstudio/scripts/param_parser.py $array_param_path -s
)
if [[ $array_param_count -gt $SLURM_ARRAY_TASK_MAX ]]; then
    echo "Array parameter count $array_param_count is greater than SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_MAX - exiting"
    exit 1
fi

array_params=$(
    $execute python3.10 nerfstudio/scripts/param_parser.py $array_param_path -i $task_id
)
if [[ -z ${array_params} ]]; then 
    echo "No array parameters found for task $task_id - exiting"
    exit 1
fi

array_params_data=$(
    $execute python3.10 nerfstudio/scripts/param_parser.py $array_param_data_path -i $data_task_id     
)
if [[ -z ${array_params_data} ]]; then 
    echo "No array parameters found for task $data_task_id - exiting"
    exit 1
fi

note=$(
    $execute python3.10 nerfstudio/scripts/param_reader.py --pipeline.note $array_params
)

output_dir=${OUTPUT_DIR:="outputs/$name"}
mkdir -p $output_dir

if [ "$dataset" == "zod" ]; then
    dataset_root="/staging/dataset_donation/round_2"
elif [ "$dataset" == "nuscenes" ]; then
    dataset_root="/datasets/nuscenes/v1.0"
elif [ "$dataset" == "pandaset" ]; then
    dataset_root="/staging/agp/datasets/pandaset"
else
    echo "Dataset must be either zod or nuscenes or pandaset, got $dataset"
    exit 1
fi

echo "Starting training with job_id $job_id"
echo "Output directory: $output_dir"
echo "Dataset root: $dataset_root"
echo "Array parameters: $array_params"
echo "Array parameters data: $array_params_data"
echo "Method: $method"

experiment_name=${EXPERIMENT_NAME:-$(date +%Y%m%d_%H%M%S)_$job_id}
experiment_name=${experiment_name}_$note

nerf_checkpoint_path=${NERF_CHECKPOINT_PATH:-"outputs/train_diffusionnerf/20240901_000000_358156_base-neurad/diffusion-nerf/2024-09-01_145604/nerfstudio_models/step-000020000.ckpt"}
if [ -z ${nerf_checkpoint_path} ]; then
    checkpoint_cmd=""
else
    checkpoint_cmd="--pipeline.nerf_checkpoint $nerf_checkpoint_path"
fi

augment_strategy=${AUGMENT_STRATEGY:-"partial_linear"}


$execute python3.10 -u nerfstudio/scripts/train.py \
    $method \
    --output-dir $output_dir \
    --vis wandb \
    --experiment-name $experiment_name \
    $checkpoint_cmd \
    --pipeline.diffusion_model.dtype "fp16" \
    --pipeline.augment_strategy $augment_strategy \
    $array_params \
    $@ \
    ${dataset}-data \
    --data $dataset_root \
    $array_params_data
    $DATAPARSER_ARGS

chmod 775 -R $output_dir

#
#EOF