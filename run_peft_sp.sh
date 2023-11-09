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
#SBATCH -o ./stdout/BestLora_3B.out  # give the job output a custom name
#SBATCH --exclude=g003,g014,g007
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
DATASET="data/small_data/small_dataset_30.fasta"
TEST_DATASET="data/small_data/small_dataset_30.fasta"

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
NUM_END_LORA=33
NUM_LORA_RANK=8
NUM_LORA_ALPHA=8

# IA3
NUM_END_IA3=0

# FINETUNE, PROMPT, TEST
TRAINING_MODE="TEST"


echo $TRAINING_MODE

if [ $TRAINING_MODE = "FINETUNE" ]; then
  echo "FINETUNE"

elif [ $TRAINING_MODE = "PROMPT" ]; then
  echo "PROMPT"

elif [ $TRAINING_MODE = "TEST" ]; then
  ### Single

parallel -j 1  \
  python scripts/train.py --data $DATASET \
  --test_partition {1} \
  --validation_partition {2} \
  --output_dir testruns/$SERIES_NAME \
  --experiment_name $EXPERIMENT_NAME \
  --model_architecture $MODEL_NAME \
  --remove_top_layers 1 \
  --sp_region_labels \
  --region_regularization_alpha 0.5 \
  --constrain_crf \
  --average_per_kingdom \
  --batch_size 20 \
  --epochs 3 \
  --optimizer adamax \
  --lr $LEARNING_RATE \
  --freeze_backbone \
  --prompt_len $PROMPT_LEN \
  --prompt_method $PROMPT_METHOD \
  --res_mlp_bottleneck_size $NUM_BOTTLENECK_SIZE \
  --num_end_adapter_layers $NUM_END_ADAPTER \
  --num_end_lora_layers $NUM_END_LORA \
  --num_lora_r $NUM_LORA_RANK \
  --num_lora_alpha $NUM_LORA_ALPHA \
::: {0..2} ::: {0..2}

else
  echo $TRAINING_MODE
  exit 1
fi