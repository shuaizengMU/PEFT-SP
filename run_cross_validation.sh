#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH -p gpu
#SBATCH -t 0-21:00
#SBATCH -N 1  # number of nodes
#SBATCH -n 8  # number of cores (AKA tasks)
#SBATCH --mem=80G
#
## labels and outputs
#SBATCH -J MF  # give the job a custom name
#SBATCH -o ./stdout/BestLora_3B_CV.out  # give the job output a custom name
#SBATCH --exclude=g003
#
#-------------------------------------------------------------------------------

source ~/data/anaconda3/bin/activate ~/data/anaconda3/envs/venv_pl

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# esm2_t48_15B_UR50D
# esm2_t36_3B_UR50D
# esm2_t33_650M_UR50D
# esm2_t30_150M_UR50D
MODEL_NAME="esm2_t30_150M_UR50D"
EXPERIMENT_NAME="ESM2-150M"
SERIES_NAME="BestLora"

# data/finetune_prompt/TATLIPO_PILIN/prompt_set.fasta
# data/small_data/small_dataset_30.fasta
# data/train_set.fasta
DATASET="data/train_set.fasta"
TEST_DATASET="data/train_set.fasta"

# SoftPromptAll
# SoftPromptFirst
# SoftPromptLast
# SoftPromptTopmost
# NoPrompt
PROMPT_METHOD="NoPrompt"
LEARNING_RATE=0.006721073999921042

# Prompt
NUM_END_PROMPT=0
PROMPT_LEN=0
NUM_BOTTLENECK_SIZE=0

# Apdapter
NUM_END_ADAPTER=0

# Lora
NUM_END_LORA=35
NUM_LORA_RANK=8
NUM_LORA_ALPHA=8


# FINETUNE, PROMPT, TEST
TRAINING_MODE="TEST"

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