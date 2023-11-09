"""function to make args for training and evaluation scripts.
"""
import argparse


def make_baseline_args(is_parsered=True):

  parser = argparse.ArgumentParser(description="Train Bert-CRF model")
  parser.add_argument(
      "--data",
      type=str,
      default="data/data/train_set.fasta",
      help="location of the data corpus. Expects test, train and valid .fasta",
  )
  parser.add_argument(
      "--sample_weights",
      type=str,
      default=None,
      help="path to .csv file with the weights for each sample",
  )
  parser.add_argument(
      "--test_partition",
      type=int,
      default=0,
      help="partition that will not be used in this training run",
  )
  parser.add_argument(
      "--validation_partition",
      type=int,
      default=1,
      help="partition that will be used for validation in this training run",
  )

  # args relating to training strategy.
  parser.add_argument("--lr", type=float, default=10,
                      help="initial learning rate")
  parser.add_argument("--clip", type=float, default=0.25,
                      help="gradient clipping")
  parser.add_argument("--epochs", type=int, default=8000,
                      help="upper epoch limit")

  parser.add_argument(
      "--batch_size", type=int, default=80, metavar="N", help="batch size"
  )
  parser.add_argument(
      "--wdecay",
      type=float,
      default=1.2e-6,
      help="weight decay applied to all weights",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="sgd",
      help="optimizer to use (sgd, adam, adamax)",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="training_run",
      help="path to save logs and trained model",
  )
  parser.add_argument(
      "--resume",
      type=str,
      default="Rostlab/prot_bert",
      help=("path of model to resume "
            "(directory containing .bin and config.json, or HF model)"),
  )
  parser.add_argument(
      "--experiment_name",
      type=str,
      default="BERT-CRF",
      help="experiment name for logging",
  )
  parser.add_argument(
      "--crossval_run",
      action="store_true",
      help=("override name with timestamp, save with split identifiers. "
            "Use when making checkpoints for crossvalidation."),
  )
  parser.add_argument(
      "--log_all_final_metrics",
      action="store_true",
      help="log all final test/val metrics to w&b",
  )

  parser.add_argument("--num_seq_labels", type=int, default=37)
  parser.add_argument("--num_global_labels", type=int, default=6)
  parser.add_argument(
      "--global_label_as_input",
      action="store_true",
      help=("Add the global label to the input sequence "
            "(only predict CS given a known label)"),
  )

  parser.add_argument(
      "--region_regularization_alpha",
      type=float,
      default=0,
      help=("multiplication factor for "
            "the region similarity regularization term"),
  )
  parser.add_argument(
      "--lm_output_dropout",
      type=float,
      default=0.1,
      help="dropout applied to LM output",
  )
  parser.add_argument(
      "--lm_output_position_dropout",
      type=float,
      default=0.1,
      help=("dropout applied to LM output, "
            "drops full hidden states from sequence"),
  )
  parser.add_argument(
      "--use_sample_weights",
      action="store_true",
      help="Use sample weights to rescale loss per sample",
  )
  parser.add_argument(
      "--use_random_weighted_sampling",
      action="store_true",
      help=("use sample weights to load random samples "
            "as minibatches according to weights"),
  )
  parser.add_argument(
      "--positive_samples_weight",
      type=float,
      default=None,
      help=("Scaling factor for positive samples loss, "
            "e.g. 1.5. Needs --use_sample_weights flag in addition."),
  )
  parser.add_argument(
      "--average_per_kingdom",
      action="store_true",
      help="Average MCCs per kingdom instead of overall computatition",
  )
  parser.add_argument(
      "--crf_scaling_factor",
      type=float,
      default=1.0,
      help="Scale CRF NLL by this before adding to global label loss",
  )
  parser.add_argument(
      "--use_weighted_kingdom_sampling",
      action="store_true",
      help="upsample all kingdoms to equal probabilities",
  )
  parser.add_argument(
      "--random_seed", type=int, default=None, help="random seed for torch."
  )
  parser.add_argument(
      "--additional_train_set",
      type=str,
      default=None,
      help="Additional samples to train on",
  )

  # args for model architecture
  parser.add_argument(
      "--model_architecture", type=str,
      choices=["bert_prottrans",
               "esm2_t48_15B_UR50D",
               "esm2_t36_3B_UR50D",
               "esm2_t33_650M_UR50D",
               "esm2_t30_150M_UR50D",
               "esm2_t12_35M_UR50D"],
      default="esm2_t30_150M_UR50D",
      help="which model architecture the checkpoint is for",)
  parser.add_argument(
      "--remove_top_layers",
      type=int,
      default=0,
      help="How many layers to remove from the top of the LM.",
  )
  parser.add_argument(
      "--kingdom_embed_size",
      type=int,
      default=0,
      help=("If >0, embed kingdom ids to N and "
            "concatenate with LM hidden states before CRF."),
  )
  parser.add_argument(
      "--use_cs_tag",
      action="store_true",
      help="Replace last token of SP with C for cleavage site",
  )
  parser.add_argument(
      "--kingdom_as_token",
      action="store_true",
      help="Kingdom ID is first token in the sequence",
  )
  parser.add_argument(
      "--sp_region_labels",
      action="store_true",
      help="Use labels for n,h,c regions of SPs.",
  )
  parser.add_argument(
      "--constrain_crf",
      action="store_true",
      help="Constrain the transitions of the region-tagging CRF.",
  )
  parser.add_argument(
      "--finetune_backbone",
      action="store_true",
      help="Finetune the backbone model.",
  )

  parser.add_argument("--num_gpu", type=int, default=3, help="number of CPUs")
  parser.add_argument("--model_name", type=str, default=None, help="model name")
  parser.add_argument("--output_root_dir", type=str, default="testruns/Prompt/",
                      help="output root")

  if is_parsered:
    args = parser.parse_args()
    return args
  else:
    return parser


def peft_sp_training_args():
  parser = make_baseline_args(is_parsered=False)

  parser.add_argument(
      "--esm2_pretrain_local", type=str, default=None,
      help="path to local esm2 pretrain model",
  )
  parser.add_argument(
      "--load", action="store_true",
      help="load S-PLM model from splm_pretrain",
  )
  parser.add_argument(
      "--splm_pretrain", type=str, default=None,
      help="path to splm pretrain model",
  )
  parser.add_argument(
      "--unfix_last_layer", type=int, default=0,
      help="unfix last layer number",
  )
  parser.add_argument(
      "--max_length", type=int, default=70,
      help="max length of sequence",
  )
  parser.add_argument(
      "--prompt_len", type=int, default=0,
      help="prompt length",
  )
  parser.add_argument(
      "--freeze_backbone", action="store_true",
      help="freeze backbone model",
  )
  parser.add_argument(
      "--prompt_method", type=str, default="NoPrompt",
      choices=["SoftPromptAll",
               "SoftPromptFirst",
               "SoftPromptLast",
               "SoftPromptTopmost",
               "NoPrompt"],
      help="prompt method",
  )
  parser.add_argument(
      "--num_end_adapter_layers", type=int, default=0,
      help="number of end adapter layers.",
  )
  parser.add_argument(
      "--res_mlp_bottleneck_size", type=int, default=0,
      help="res mlp bottleneck size.",
  )
  parser.add_argument("--use_adapter", action="store_true",
                      help="use adapter")
  parser.add_argument("--optuna_use_lora", action="store_true",
                      help="use lora")
  parser.add_argument("--optuna_use_ia3", action="store_true",
                      help="use ia3")
  parser.add_argument("--use_embedding_encoder", action="store_true",
                      help="use embedding encoder")
  parser.add_argument("--num_end_prompt_layers", type=int,
                      help="number of end prompt layers.")
  parser.add_argument("--num_end_lora_layers", type=int,
                      help="number of end lora layers.")
  parser.add_argument("--num_lora_r", type=int,
                      help="value of rank (r) in lora.")
  parser.add_argument("--num_lora_alpha", type=int,
                      help="value of alpha in lora.")
  parser.add_argument("--num_end_ia3_layers", type=int,
                      help="number of end ia3 layers.")

  args = parser.parse_args()
  return args


def cross_validate_args():

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data",
      type=str,
  )
  parser.add_argument(
      "--model_base_path",
      type=str,
      default="/work3/felteu/tagging_checkpoints/signalp_6",
  )
  parser.add_argument("--randomize_kingdoms", action="store_true")
  parser.add_argument("--output_file", type=str,
                      default="crossval_metrics.csv")
  parser.add_argument(
      "--no_multistate",
      action="store_true",
      help=("Model to evaluate is no multistate model, "
            "use different labels to find CS"),
  )
  parser.add_argument(
      "--n_partitions",
      type=int,
      default=5,
      help="Number of partitions, for loading the checkpoints and datasets.",
  )
  parser.add_argument(
      "--use_pvd",
      action="store_true",
      help="Replace viterbi decoding with posterior-viterbi decoding",
  )
  parser.add_argument(
      "--model_architecture", type=str,
      choices=["bert_prottrans",
               "esm2_t48_15B_UR50D",
               "esm2_t36_3B_UR50D",
               "esm2_t33_650M_UR50D",
               "esm2_t30_150M_UR50D",
               "esm2_t12_35M_UR50D"],
      default=None,
      help="which model architecture the checkpoint is for",
  )
  parser.add_argument(
      "--esm2_pretrain_local", type=str, default=None,
      help="path to local esm2 pretrain model",
  )
  parser.add_argument(
      "--load", action="store_true",
      help="load S-PLM model from splm_pretrain",
  )
  parser.add_argument(
      "--splm_pretrain", type=str, default=None,
      help="path to splm pretrain model",
  )
  parser.add_argument(
      "--unfix_last_layer", type=int, default=0,
      help="unfix last layer number",
  )
  parser.add_argument(
      "--max_length", type=int, default=70,
      help="max length of sequence",
  )
  parser.add_argument(
      "--prompt_len", type=int, default=0,
      help="prompt length",
  )
  parser.add_argument(
      "--freeze_backbone", action="store_true",
      help="freeze backbone model",
  )
  parser.add_argument("--lm_output_dropout",
                      type=float, default=0.1,
                      help="dropout applied to LM output",)
  parser.add_argument("--num_seq_labels", type=int, default=37)
  parser.add_argument("--num_global_labels", type=int, default=6)
  parser.add_argument(
      "--lm_output_position_dropout",
      type=float,
      default=0.1,
      help=("dropout applied to LM output, "
            "drops full hidden states from sequence"),
  )
  parser.add_argument(
      "--crf_scaling_factor",
      type=float,
      default=1.0,
      help="Scale CRF NLL by this before adding to global label loss",
  )
  parser.add_argument(
      "--sp_region_labels", action="store_true",
      help="",
  )
  parser.add_argument(
      "--kingdom_embed_size",
      type=int,
      default=0,
      help=("If >0, embed kingdom ids to N and "
            "concatenate with LM hidden states before CRF."),
  )
  parser.add_argument(
      "--constrain_crf",
      action="store_true",
      help="Constrain the transitions of the region-tagging CRF.",
  )
  parser.add_argument(
      "--kingdom_as_token",
      action="store_true",
      help="Kingdom ID is first token in the sequence",
  )
  parser.add_argument(
      "--average_per_kingdom",
      action="store_true",
      help="Average MCCs per kingdom instead of overall computatition",
  )
  parser.add_argument(
      "--global_label_as_input",
      action="store_true",
      help=("Add the global label to the input sequence "
            "(only predict CS given a known label)"),
  )
  parser.add_argument(
      "--prompt_method", type=str, default="NoPrompt",
      choices=["SoftPromptAll",
               "SoftPromptFirst",
               "SoftPromptLast",
               "SoftPromptTopmost",
               "NoPrompt"],
      help="prompt method",
  )
  parser.add_argument(
      "--num_end_adapter_layers", type=int, default=0,
      help="number of end adapter layers.",
  )
  parser.add_argument(
      "--res_mlp_bottleneck_size", type=int, default=0,
      help="res mlp bottleneck size.",
  )
  parser.add_argument("--use_adapter", action="store_true",
                      help="use adapter")
  parser.add_argument("--optuna_use_lora", action="store_true",
                      help="use lora")
  parser.add_argument("--optuna_use_ia3", action="store_true",
                      help="use ia3")
  parser.add_argument("--use_embedding_encoder", action="store_true",
                      help="use embedding encoder")
  parser.add_argument("--num_end_prompt_layers", type=int,
                      help="number of end prompt layers.")
  parser.add_argument("--num_end_lora_layers", type=int,
                      help="number of end lora layers.")
  parser.add_argument("--num_lora_r", type=int,
                      help="value of rank (r) in lora.")
  parser.add_argument("--num_lora_alpha", type=int,
                      help="value of alpha in lora.")
  parser.add_argument("--num_end_ia3_layers", type=int,
                      help="number of end ia3 layers.")

  args = parser.parse_args()
  return args
