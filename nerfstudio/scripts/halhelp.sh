#!/bin/bash
#SBATCH --nodes 1
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/neurad_imaginedriving/slurm/%j_%a.out
#SBATCH --job-name=neurad_imaginedriving
#SBATCH --partition zprodlow


image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/neuraddiffusion-03_05_24.sif"}
runscript=${RUNSCRIPT:-"train.py"}
method=${METHOD:-imaginedriving}

singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /staging:/staging \
    --bind /workspaces:/workspaces \
    --bind /datasets:/datasets \
    $image_path \
    python3.10 nerfstudio/scripts/$runscript \
    ${@:1} --help
#
#EOF