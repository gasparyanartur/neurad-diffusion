#!/bin/bash

#SBATCH --job-name=render
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 100G
#SBATCH --output logs/%x/slurm/%j.out
#SBATCH --array=0
#SBATCH --time 2-00:00:00
#SBATCH --account berzelius-2024-347

source .env

name=${NAME:-"viewer"}

image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}
if [ -z ${NERF_CONFIG_PATH} ]; then
    echo "NERF_CONFIG_PATH not set. Exiting."
    exit 1;
fi


#BIND_CMD="--bind /staging:/staging --bind /workspaces:/workspaces --bind /datasets:/datasets"
BIND_CMD="--bind /proj:/proj --bind /home:/home"

execute="singularity exec \
            --nodes 1 \
            --gpus 1 \
            --cpus-per-task 32 \
            --mem 100G \
            --nv \
            $BIND_CMD \
            --home /nerfstudio \
            --bind $PWD:/nerfstudio \
            --env PYTHONPATH=/nerfstudio \
            --env WANDB_API_KEY=$WANDB_API_KEY \
            $image_path"

$execute python3.10 -u nerfstudio/scripts/viewer/run_viewer.py \
    --load-config $NERF_CONFIG_PATH \
    $@ 


#
#EOF