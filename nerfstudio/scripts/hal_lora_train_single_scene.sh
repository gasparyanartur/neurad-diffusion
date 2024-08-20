!/bin/bash

SBATCH --nodes 1
SBATCH --gpus 1
SBATCH --cpus-per-task 8
SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/finetune/slurm/%j.out
SBATCH --partition=zprod 
SBATCH --job-name=finetune

if [ -z ${WANDB_API_KEY} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi


lora_train_config=${LORA_TRAIN_CONFIG:-"configs/hal-configs/train_model.yml"}
image_path=${IMAGE_PATH:-"containers/neuraddiffusion-24_05_24.sif"}

main_process_port=${MAIN_PROCESS_PORT:-29500}
num_processes=${NUM_PROCESSES:-1}
num_machines=${NUM_MACHINES:-1}
dynamo_backend=${DYNAMO_BACKEND:-"no"}
mixed_precision=${MIXED_PRECISION:-"no"}

logging_dir=${LOGGING_DIR:-"logs/finetune/slurm"}
output_dir=${OUTPUT_DIR:-"outputs/finetune"}

dataset_dir=${DATASET_DIR:-"data/pandaset"}
nerf_output_dir=${NERF_OUTPUT_DIR:-"data/nerf_outputs"}

train_batch_size=${TRAIN_BATCH_SIZE:-"1"}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-"4"}

diffusion_type=${DIFFUSION_TYPE:-"sd"}

#singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --num_processes=$num_processes --num_machines=$num_machines --dynamo_backend=$dynamo_backend --mixed_precision=$mixed_precision --main_process_port=$main_process_port scripts/run_train_lora.py $config_path
singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /home:/home \
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
        --logging_dir $logging_dir \
        --output_dir $output_dir \
        --dataset_config.type "single_scene" \
        --dataset_config.dataset_dir $dataset_dir \
        --dataset_config.nerf_output_dir $nerf_output_dir \
        --train_batch_size $train_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --diffusion.type $diffusion_type \
        ${@:1} 