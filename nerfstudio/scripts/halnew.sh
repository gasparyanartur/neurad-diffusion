#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/neurad_imaginedriving/%j_%a.out
#SBATCH --array=1-10
#SBATCH --job-name=neurad_imaginedriving
#SBATCH --partition zprodlow



# crash if no argument is given
name=${1:?"No name given"}

wandb_api_key=${WANDB_API_KEY}
if [ -z ${wandb_api_key} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

method=${METHOD:-neurad}
dataset=${DATASET:-pandaset}
# Specify the path to the config file
id_to_seq=nerfstudio/scripts/arrays/${dataset}_id_to_seq${SUFFIX}.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
[[ -z $seq ]] && exit 1

# For each sequence, start the training
echo "Starting training for $name with extra args ${@:2}"
echo "Sequence $seq"

export OUTPUT_DIR=${OUTPUT_DIR:="/staging/agp/masterthesis/nerf-thesis-shared/output/$DATASET-$METHOD"}
output_dir=${OUTPUT_DIR:="outputs/$dataset-$method"}
mkdir -p $output_dir


image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/neurad_25_04_24.sif"}


if [ -z ${LOAD_NAME+x} ]; then
    MAYBE_RESUME_CMD=""
else
    echo "LOAD_NAME specified in environment, resuming from $LOAD_NAME"
    checkpoints=( $(ls outputs/$LOAD_NAME-$seq/$method/*/nerfstudio_models/*.ckpt) )
    MAYBE_RESUME_CMD="--load-checkpoint=${checkpoints[-1]}"
fi


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

singularity exec --nv \
    --bind /staging:/staging \
    --bind /workspaces:/workspaces \
    --bind /datasets:/datasets \
    --env WANDB_API_KEY=$wandb_api_key \
    $image_path \
    python3.10 -u nerfstudio/scripts/train.py \
    $method \
    --output-dir $output_dir \
    --vis wandb \
    --experiment-name $name-$seq \
    $MAYBE_RESUME_CMD \
    ${@:2} \
    ${dataset}-data \
    --data $dataset_root \
    --sequence $seq \
    $DATAPARSER_ARGS
#
#EOF