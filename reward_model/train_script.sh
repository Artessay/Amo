
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct" # model path

TRAIN_DATASETS="../assets/text_to_text/preference" # rm dataset path
TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="$../outputs/qwen_2_5_rm" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --epochs 1 