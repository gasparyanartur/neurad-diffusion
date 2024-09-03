#!/bin/bash

#SBATCH --job-name=render
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 100G
#SBATCH --output logs/%x/slurm/%j.out
#SBATCH --array=0
#SBATCH --time 2-00:00:00

source .env
if [ -z ${WANDB_API_KEY} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

export WANDB_RUN_GROUP=$name
export WANDB_ENTITY=arturruiqi
export WANDB_PROJECT=diffusionnerf

name=${NAME:-"render"}
subcommand=${SUBCOMMAND:-dataset}
cameras=${CAMERAS:-front}
output_dir=${OUTPUT_DIR:-"renders"}

job_id=${SLURM_JOB_ID:-"000000"}
task_id=${SLURM_ARRAY_TASK_ID:-"1"}
image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}

array_param_path=${ARRAY_PARAM_PATH:-nerfstudio/scripts/params/${name}.json}

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

x_shift=$(
    $execute python3.10 nerfstudio/scripts/param_reader.py --x-shift $array_params
)
run_name=$(
    $execute python3.10 nerfstudio/scripts/param_reader.py --run-name $array_params
)
load_config=$(
    $execute python3.10 nerfstudio/scripts/param_reader.py --load-config $array_params
)
if [[ -z ${load_config} ]]; then 
    echo "No load config found for task $task_id - exiting"
    exit 1
fi
load_config_parent=$(dirname "$load_config")
load_config_name=$(basename "$load_config")
if [[ $load_config_name != "config.yml"]]; then 
    echo "Load config must be named config.yml - exiting"
    exit 1
fi

scene=$(basename "$load_config_parent")
if [[ -z ${scene} ]]; then 
    echo "No scene found for task $task_id - exiting"
    exit 1
fi
if [[ "$scene" =~ ^\d{3}$ ]]; then 
    echo "expected scene to be parent of config - exiting"
    exit 1
fi

scene_parent=$(dirname "$load_config_parent")
model_name=$(basename "$scene_parent")


if [[ -z ${x_shift} ]]; then 
    echo "No x-shift found for task $task_id - exiting"
    exit 1
fi


output_path=$output_dir/$model_name/$scene/$cameras/${x_shift}m/${job_id}_${run_name}
mkdir -p $output_path

echo "Starting renderings with job_id ${job_id}"
echo "Subcommand: $subcommand"
echo "Output path: $output_path"
echo "Array parameters: $array_params"
echo "Cameras: $cameras"

$execute python3.10 -u nerfstudio/scripts/render.py \
    $subcommand \
    --pose-source "train+test" \
    --calculate_and_save_metrics True \
    --cameras $cameras \
    --output-path $output_path \
    --shift $x_shift 0 0 \
    --load_config $load_config \
    $@ 

chmod 775 -R $output_path

#
#EOF