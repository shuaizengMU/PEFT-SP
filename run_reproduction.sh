#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH -p gpu
#SBATCH -t 0-21:00
#SBATCH -N 1  # number of nodes
#SBATCH -n 8  # number of cores (AKA tasks)
#SBATCH --mem=120G
#
## labels and outputs
#SBATCH -J MF  # give the job a custom name
#SBATCH -o ./stdout/BestLoraCls4_ESM650.out  # give the job output a custom name
#SBATCH --exclude=g003
#
#-------------------------------------------------------------------------------

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="0,1,2,3"



# data/small_data/small_dataset_30.fasta
# data/train_set.fasta
DATASET="data/small_data/small_dataset_30.fasta"
TEST_DATASET="data/small_data/small_dataset_30.fasta"


# esm2_t48_15B_UR50D
# esm2_t36_3B_UR50D
# esm2_t33_650M_UR50D
# esm2_t30_150M_UR50D
MODEL_NAME="esm2_t30_150M_UR50D"
EXPERIMENT_NAME="ESM2-150M"
SERIES_NAME="BestLora"

# PROMPT_METHOD="NoPrompt"
# LEARNING_RATE=0.006950570420823288

# # Prompt
# NUM_END_PROMPT=0
# PROMPT_LEN=0
# NUM_BOTTLENECK_SIZE=0

# # Apdapter
# NUM_END_ADAPTER=0

# # Lora
# NUM_END_LORA=25
# NUM_LORA_RANK=8
# NUM_LORA_ALPHA=8


# # FINETUNE, PROMPT, TEST
# TRAINING_MODE="TEST"


### LoRA
MODEL_NAME="esm2_t30_150M_UR50D"
EXPERIMENT_NAME="ESM2-150M"
SERIES_NAME="BestLora"

PROMPT_METHOD="NoPrompt"
LEARNING_RATE=0.006950570420823288
NUM_END_PROMPT=0
PROMPT_LEN=0
NUM_BOTTLENECK_SIZE=0
NUM_END_ADAPTER=0
NUM_END_LORA=25
NUM_LORA_RANK=8
NUM_LORA_ALPHA=8

python scripts/cross_validate.py \
--data $TEST_DATASET \
--model_base_path testruns/$SERIES_NAME/$EXPERIMENT_NAME \
--n_partitions 3 \
--output_file testruns/$SERIES_NAME/$EXPERIMENT_NAME/crossval_metrics.csv \
--model_architecture $MODEL_NAME \
--constrain_crf \
--average_per_kingdom \
--sp_region_labels \
--prompt_method $PROMPT_METHOD \
--prompt_len $PROMPT_LEN \
--num_end_lora_layers $NUM_END_LORA \
--num_lora_r $NUM_LORA_RANK \
--num_lora_alpha $NUM_LORA_ALPHA 
