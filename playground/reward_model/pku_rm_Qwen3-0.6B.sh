
MODEL_NAME_OR_PATH="/data/Qwen/Qwen3-0.6B" # model path

TRAIN_DATASETS="./PKU-SafeRLHF" # rm dataset path
TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="./outputs/Qwen3-0.6B-SafeRLHF-RM" # output dir

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
     --per_device_train_batch_size 64 \
     --per_device_eval_batch_size 64 \
     --epochs 1 