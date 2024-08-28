config_neurad="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331981/imaginedriving-001-331981-1/imaginedriving/2024-05-25_130746/config.yml"
config_diffusion="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331982/imaginedriving-001-331982-1/imaginedriving/2024-05-25_130744/config.yml"
config_diffusion_lora="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/332043/imaginedriving-001-332043-1/imaginedriving/2024-05-26_174218/config.yml"

shift1="0 0 0"
shift2="2 0 0"
shift3="4 0 0"
shift4="6 0 0"
shift5="8 0 0"

output_base=/staging/agp/masterthesis/nerf-thesis-shared/renders/neurad_imaginedriving/pandaset-imaginedriving

CONFIGPATH=$config_neurad OUTPUTPATH="${output_base}/config_neurad/0m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift1
CONFIGPATH=$config_neurad OUTPUTPATH="${output_base}/config_neurad/2m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift2
CONFIGPATH=$config_neurad OUTPUTPATH="${output_base}/config_neurad/4m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift3
CONFIGPATH=$config_neurad OUTPUTPATH="${output_base}/config_neurad/6m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift4
CONFIGPATH=$config_neurad OUTPUTPATH="${output_base}/config_neurad/8m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift5

CONFIGPATH=$config_diffusion OUTPUTPATH="${output_base}/config_diffusion/0m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift1
CONFIGPATH=$config_diffusion OUTPUTPATH="${output_base}/config_diffusion/2m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift2
CONFIGPATH=$config_diffusion OUTPUTPATH="${output_base}/config_diffusion/4m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift3
CONFIGPATH=$config_diffusion OUTPUTPATH="${output_base}/config_diffusion/6m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift4
CONFIGPATH=$config_diffusion OUTPUTPATH="${output_base}/config_diffusion/8m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift5

CONFIGPATH=$config_diffusion_lora OUTPUTPATH="${output_base}/config_diffusion_lora/0m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift1
CONFIGPATH=$config_diffusion_lora OUTPUTPATH="${output_base}/config_diffusion_lora/2m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift2
CONFIGPATH=$config_diffusion_lora OUTPUTPATH="${output_base}/config_diffusion_lora/4m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift3
CONFIGPATH=$config_diffusion_lora OUTPUTPATH="${output_base}/config_diffusion_lora/6m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift4
CONFIGPATH=$config_diffusion_lora OUTPUTPATH="${output_base}/config_diffusion_lora/8m" sbatch nerfstudio/scripts/halnew_render.sh --shift $shift5
