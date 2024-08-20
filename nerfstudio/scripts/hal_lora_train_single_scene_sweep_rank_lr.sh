# Sweep across a set of parameters

# overhead
lora_train_config=${LORA_TRAIN_CONFIG:-"configs/hal-configs/train_lora.yml"}
image_path=${IMAGE_PATH:-"containers/neurad_140824.sif"}

main_process_port=${MAIN_PROCESS_PORT:-29500}
num_processes=${NUM_PROCESSES:-1}
num_machines=${NUM_MACHINES:-1}
dynamo_backend=${DYNAMO_BACKEND:-"no"}
mixed_precision=${MIXED_PRECISION:-"no"}


# parameters
lora_ranks=("4" "8" "16" "32" "64" "128")
learning_rates=("0.0001" "0.0002" "0.0003")

# fixed parameters
noise_strength="0.1"
diffusion_type="sd"


for lora_rank in ${lora_ranks[@]}; do
    for learning_rate in ${learning_rates[@]}; do
        LORA_TRAIN_CONFIG=$lora_train_config \
        IMAGE_PATH=$image_path \
        MAIN_PROCESS_PORT=$main_process_port \
        NUM_PROCESSES=$num_processes \
        NUM_MACHINES=$num_machines \
        DYNAMO_BACKEND=$dynamo_backend \
        MIXED_PRECISION=$mixed_precision \
        WANDB_API_KEY=$WANDB_API_KEY \
        sbatch --partition=zprodlow nerfstudio/scripts/hal_finetune.sh \
            --diffusion_config.type $diffusion_type \
            --diffusion_config.noise_strength $noise_strength \
            --diffusion_config $lora_rank \
            --noise_strength $noise_strength \
            --learning_rate $learning_rate
    done
done