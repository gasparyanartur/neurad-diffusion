#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH --time 0-05:00:00
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/neurad_imaginedriving/slurm/%j.out
#SBATCH --partition zprodlow
#SBATCH --array=1-2
#SBATCH --job-name=neurad_imaginedriving

export METHOD=${METHOD:-neurad}
export DATASET=${DATASET:-pandaset}
export OUTPUT_DIR=${OUTPUT_DIR:="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/$DATASET-$METHOD/$SLURM_JOB_ID"}
export WANDB_RUN_GROUP=$name
export WANDB_ENTITY=${WANDB_ENTITY:-arturruiqi}
export WANDB_PROJECT=${WANDB_PROJECT:-neurad_imaginedriving} 

if [ "$DATASET" == "zod" ]; then
    DATASET_ROOT="/staging/dataset_donation/round_2"
elif [ "$DATASET" == "nuscenes" ]; then
    DATASET_ROOT="/datasets/nuscenes/v1.0"
elif [ "$DATASET" == "pandaset" ]; then
    DATASET_ROOT="/staging/agp/datasets/pandaset"
else
    echo "Dataset must be either zod or nuscenes or pandaset, got $DATASET"
    exit 1
fi

export DATASET_ROOT=$DATASET_ROOT
image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/neurad_24_04_24.sif"}

export SINGULARITY_CMD="--bind /staging:/staging --bind /workspaces:/workspaces --bind /datasets:/datasets $image_path"

scripts/_run_array.sh $@




