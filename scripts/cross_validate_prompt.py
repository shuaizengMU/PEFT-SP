# © Copyright Technical University of Denmark
"""
Compute cross-validated metrics. Run each model
on its test set, compute all metrics and save as .csv
We save all metrics per model in the .csv, mean/sd
calculation we do later in a notebook.
"""
import torch
import os
import argparse

from signalp6.models import (ProteinBertTokenizer, BertSequenceTaggingCRF,
                             ESMSequencePromptCRF)
from signalp6.training_utils import RegionCRFDataset, ESM2CRFDataset
from signalp6.utils import get_metrics_multistate
from tqdm import tqdm
import pandas as pd

from transformers import BertConfig, EsmConfig

from duolin_code import duolin_utilities, prompts

import config_utilities
import args_maker

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, IA3Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data",
    #     type=str,
    #     default="../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta",
    # )
    # parser.add_argument(
    #     "--model_base_path",
    #     type=str,
    #     default="/work3/felteu/tagging_checkpoints/signalp_6",
    # )
    # parser.add_argument("--randomize_kingdoms", action="store_true")
    # parser.add_argument("--output_file", type=str,
    #                     default="crossval_metrics.csv")
    # parser.add_argument(
    #     "--no_multistate",
    #     action="store_true",
    #     help="Model to evaluate is no multistate model, use different labels to find CS",
    # )
    # parser.add_argument(
    #     "--n_partitions",
    #     type=int,
    #     default=5,
    #     help="Number of partitions, for loading the checkpoints and datasets.",
    # )
    # parser.add_argument(
    #     "--use_pvd",
    #     action="store_true",
    #     help="Replace viterbi decoding with posterior-viterbi decoding",
    # )
    # parser.add_argument(
    #   "--model_architecture", type=str,
    #   choices=["bert_prottrans",
    #            "esm2_t48_15B_UR50D",
    #            "esm2_t36_3B_UR50D",
    #            "esm2_t33_650M_UR50D",
    #            "esm2_t30_150M_UR50D",
    #            "esm2_t12_35M_UR50D"],
    #   default="esm2_t30_150M_UR50D",
    #   help="which model architecture the checkpoint is for",
    # )
    # parser.add_argument(
    #     "--esm2_pretrain_local", type=str, default=None,
    #     help="path to local esm2 pretrain model",
    # )
    # parser.add_argument(
    #     "--load", action="store_true",
    #     help="load S-PLM model from splm_pretrain",
    # )
    # parser.add_argument(
    #     "--splm_pretrain", type=str, default=None,
    #     help="path to splm pretrain model",
    # )
    # parser.add_argument(
    #     "--unfix_last_layer", type=int, default=0,
    #     help="unfix last layer number",
    # )
    # parser.add_argument(
    #     "--max_length", type=int, default=70,
    #     help="max length of sequence",
    # )
    # parser.add_argument(
    #     "--prompt_len", type=int, default=0,
    #     help="prompt length",
    # )
    # parser.add_argument(
    #     "--freeze_backbone", action="store_true",
    #     help="freeze backbone model",
    # )
    # parser.add_argument("--lm_output_dropout",
    #                     type=float,default=0.1,
    #                     help="dropout applied to LM output",)
    # parser.add_argument("--num_seq_labels", type=int, default=37)
    # parser.add_argument("--num_global_labels", type=int, default=6)
    # parser.add_argument(
    #   "--lm_output_position_dropout",
    #   type=float,
    #   default=0.1,
    #   help="dropout applied to LM output, drops full hidden states from sequence",
    #   )
    # parser.add_argument(
    #   "--crf_scaling_factor",
    #   type=float,
    #   default=1.0,
    #   help="Scale CRF NLL by this before adding to global label loss",
    # )
    # parser.add_argument(
    #     "--sp_region_labels", action="store_true",
    #     help="",
    # )
    # parser.add_argument(
    #   "--kingdom_embed_size",
    #   type=int,
    #   default=0,
    #   help="If >0, embed kingdom ids to N and concatenate with LM hidden states before CRF.",
    #   )
    # parser.add_argument(
    #   "--constrain_crf",
    #   action="store_true",
    #   help="Constrain the transitions of the region-tagging CRF.",
    #   )
    # parser.add_argument(
    #   "--kingdom_as_token",
    #   action="store_true",
    #   help="Kingdom ID is first token in the sequence",
    #   )
    # parser.add_argument(
    #   "--average_per_kingdom",
    #   action="store_true",
    #   help="Average MCCs per kingdom instead of overall computatition",
    #   )
    # parser.add_argument(
    #   "--global_label_as_input",
    #   action="store_true",
    #   help="Add the global label to the input sequence (only predict CS given a known label)",
    #   )
    # parser.add_argument(
    #     "--prompt_method", type=str, default="NoPrompt",
    #     choices=["SoftPromptAll", "SoftPromptFirst", "SoftPromptLast",
    #             "NoPrompt"],
    #     help="prompt method",
    # )
    # args = parser.parse_args()
    args = args_maker.cross_validate_args()

    # tokenizer = ProteinBertTokenizer.from_pretrained(
    #     "data/tokenizer",
    #     do_lower_case=False,
    # )

    hidden_size_dict = {
        "esm2_t48_15B_UR50D": 5120,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t12_35M_UR50D": 480,
    }
    assert "esm" in args.model_architecture, "Only support esm model."

    # Sequence model
    model_seq = duolin_utilities.load_model(args).to(device)

    alphabet = model_seq.esm2.alphabet
    batch_converter = alphabet.get_batch_converter(
        truncation_seq_length=args.max_length)

    # Prompt model
    num_esm_layers = len(model_seq.esm2.layers)
    if args.prompt_method == "SoftPromptFirst":
      prompt_layer_idx_list = [0]
    elif args.prompt_method == "SoftPromptAll":
      prompt_layer_idx_list = [i for i in range(0, num_esm_layers)]
    elif args.prompt_method == "SoftPromptLast":
      prompt_layer_idx_list = [num_esm_layers-1]
    elif args.prompt_method == "SoftPromptTopmost" and args.num_end_prompt_layers > 0:
      start_layer_idx = num_esm_layers - args.num_end_prompt_layers
      prompt_layer_idx_list = [idx
                               for idx in range(
                                   start_layer_idx, num_esm_layers)]
    else:
      args.prompt_len = 0
      prompt_layer_idx_list = None

    prompt_initial_func = prompts.from_sample_of_embeddings
    x_embed_table = model_seq.esm2.embed_tokens.weight
    model_prompt = prompts.Prompts(
        args.prompt_len, prompt_initial_func, x_embed_table,
        num_esm_layers=num_esm_layers,
        prompt_layer_idx_list=prompt_layer_idx_list,
        original_seq_len=70, device=device, args=args)

    if model_prompt.res_mlp is not None:
      print(f"Using Res MLP prompt")
    else:
      print(f"No Res MLP")

    # Collect results + header info to build output df
    results_list = []

    partitions = list(range(args.n_partitions))

    # for result collection
    metrics_list = []
    checkpoint_list = []
    for partition in tqdm(partitions):
      # Load data
      dataset = ESM2CRFDataset(
          args.data,
          batch_converter=batch_converter,
          partition_id=[partition],
          add_special_tokens=True,
      )

      if args.randomize_kingdoms:
          import random

          kingdoms = list(set(dataset.kingdom_ids))
          random_kingdoms = random.choices(
              kingdoms, k=len(dataset.kingdom_ids))
          dataset.kingdom_ids = random_kingdoms
          print("randomized kingdom IDs")

      dl = torch.utils.data.DataLoader(
          dataset, collate_fn=dataset.collate_fn, batch_size=50
      )

      # Put together list of checkpoints
      checkpoints = [
          os.path.join(
              args.model_base_path, f"test_{partition}_valid_{x}", "model.pt")
          for x in set(partitions).difference({partition})]

      # Run
      print(checkpoints)
      for checkpoint in checkpoints:

        # model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        config = EsmConfig.from_pretrained("Rostlab/prot_bert")

        config = config_utilities.load_basic_config(config, args)

        setattr(config, "model_seq", model_seq)
        setattr(config, "model_prompt", model_prompt)
        setattr(
            config, "hidden_size",
            hidden_size_dict[args.model_architecture])

        model = ESMSequencePromptCRF(config)

        if args.num_end_lora_layers > 0:
          print(
              f"Using LoRA: rank {args.num_lora_r}, at last {args.num_end_lora_layers} layers.")
          target_modules = []
          start_layer_idx = num_esm_layers - args.num_end_lora_layers
          for idx in range(start_layer_idx, num_esm_layers):
            for layer_name in [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj"]:
              target_modules.append(f"layers.{idx}.{layer_name}")

          peft_config = LoraConfig(inference_mode=True,
                                    r=args.num_lora_r,
                                    lora_alpha=args.num_lora_alpha,
                                    target_modules=target_modules,
                                    lora_dropout=0.1,
                                    bias="none",)

          model = get_peft_model(model, peft_config)

        else:
          print(f"No LoRA...")


        if args.num_end_ia3_layers > 0:
          
          print(f"Using IA3: at last {args.num_end_ia3_layers} layers.")
          target_modules = []
          feedforward_modules = []
          start_layer_idx = num_esm_layers - args.num_end_ia3_layers
          for idx in range(start_layer_idx, num_esm_layers):
            for layer_name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]:
              target_modules.append(f"layers.{idx}.{layer_name}")
            feedforward_modules.append(f"layers.{idx}.self_attn.out_proj")

          peft_config = IA3Config(
            inference_mode=False, 
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
          )
          model = get_peft_model(model, peft_config)
        else:
          print(f"No IA3...")


        model.load_state_dict(torch.load(checkpoint))
        setattr(model, "use_pvd", args.use_pvd)  # ad hoc fix to use pvd

        metrics = get_metrics_multistate(
            model,
            dl,
            sp_tokens=[3, 7, 11, 15, 19] if args.no_multistate else None,
            compute_region_metrics=False, args=args
        )
        metrics_list.append(metrics)  # save metrics
        # checkpoint_list.append(f'test_{partition}_val_{x}') #save name of checkpoint

    df = pd.DataFrame.from_dict(metrics_list)
    # df.index = checkpoint_list

    df.T.to_csv(args.output_file)


if __name__ == "__main__":
    main()
