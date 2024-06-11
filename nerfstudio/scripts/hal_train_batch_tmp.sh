neurad_path="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331982/imaginedriving-001-331982-1/imaginedriving/2024-05-25_130744"
neurad_checkpoint_path="${neurad_path}/nerfstudio_models/step-000020000.ckpt"
lora_weights="/staging/agp/masterthesis/nerf-thesis-shared/output/finetune/sd/335491/final_weights/pytorch_lora_weights.safetensors"

lora_cmds=("_" " $lora_weights")

augment_strategys=("none" "partial_linear")
max_shifts=(4)
max_rots=(45 90 180)

for lora_cmd in ${lora_cmds[@]}; do
    for augment_strategy in ${augment_strategys[@]}; do
        for max_shift in ${max_shifts[@]}; do
            for max_rot in ${max_rots[@]}; do
                WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' sbatch nerfstudio/scripts/halnew.sh imaginedriving\
                    --pipeline.augment_strategy $augment_strategy\
                    --pipeline.nerf_checkpoint $neurad_checkpoint_path\
                    --pipeline.augment_phase_step 0\
                    --pipeline.diffusion_model.lora_weights $lora_weights\
                    --pipeline.diffusion_model.dtype "fp16"\
                    --pipeline.augment_max_strength $max_shift 0 0 0 0 $max_rot
            done
        done
    done
done

echo ""

#WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH='.' python3.10 nerfstudio/scripts/train.py imaginedriving --output-dir outputs/imaginedriving --vis wandb --experiment-name "test-run" --pipeline.augment_strategy $augment_strategy --pipeline.nerf_checkpoint $neurad_checkpoint_path pandaset-data --data data/pandaset --sequence 001 --cameras front 
