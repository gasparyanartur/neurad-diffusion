# crash if no argument is given
name=${1:?"No name given"}
dataset=${DATASET:?"No dataset given"}
dataset_root=${DATASET_ROOT:?"No dataset root given"}
method=${METHOD:?"neurad"}

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
id_to_seq=scripts/arrays/${dataset}_id_to_seq${SUFFIX}.txt
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
[[ -z $seq ]] && exit 1

# For each sequence, start the training
echo "Starting training for $name with extra args ${@:2}"
echo "Sequence $seq"

output_dir=${OUTPUT_DIR:="outputs/$dataset-$method"}
mkdir -p $output_dir

if [ -z ${LOAD_NAME+x} ]; then
    MAYBE_RESUME_CMD=""
else
    echo "LOAD_NAME specified in environment, resuming from $LOAD_NAME"
    checkpoints=( $(ls $output_dir/$LOAD_NAME-$seq/$method/*/nerfstudio_models/*.ckpt) )
    MAYBE_RESUME_CMD="--load-checkpoint=${checkpoints[-1]}"
fi

singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --pwd /nerfstudio \
    ${SINGULARITY_CMD:?"must specify singularity command, at least the container (.sif) path"} \
    python -u nerfstudio/scripts/train.py \
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
