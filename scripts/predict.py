""" Compute cross-validated metrics. 
Run each model on its test set, compute all metrics and save as .csv

The original code is from SignalP 6.0.
"""
import torch
import os
import pandas as pd
import random

from peft import LoraConfig, get_peft_model
from tqdm import tqdm

import args_maker
from signalp6.models import PeftSPEsmCRF
from signalp6.training_utils import ESM2CRFDataset
from signalp6.utils import get_metrics_multistate
from model_config import PeftSPConfig
from utilities import esm_utilities, prompts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.hub.set_dir("./torch_hub")


def main():

  args = args_maker.cross_validate_args()
  assert "esm" in args.model_architecture, "Only support esm model."

  # backbnone model
  model_seq = esm_utilities.load_model(args.model_architecture,
                                       args.num_end_adapter_layers,
                                       args.freeze_backbone).to(DEVICE)
  num_esm_layers = len(model_seq.esm2.layers)
  alphabet = model_seq.esm2.alphabet
  batch_converter = alphabet.get_batch_converter(
      truncation_seq_length=args.max_length)

  # prompt tuning model
  prompt_initial_func = prompts.from_sample_of_embeddings
  x_embed_table = model_seq.esm2.embed_tokens.weight
  model_prompt = prompts.Prompts(
      args.prompt_len, prompt_initial_func, x_embed_table,
      num_esm_layers=num_esm_layers,
      original_seq_len=70,
      prompt_method=args.prompt_method,
      num_end_prompt_layers=args.num_end_prompt_layers,
      res_mlp_bottleneck_size=args.res_mlp_bottleneck_size, device=DEVICE)

  if model_prompt.res_mlp is not None:
    print("Using Res MLP prompt")
  else:
    print("No Res MLP")

  config = PeftSPConfig(args.num_seq_labels, args.num_global_labels,
                        args.lm_output_dropout, args.lm_output_position_dropout,
                        args.crf_scaling_factor, args.sp_region_labels,
                        args.kingdom_embed_size, args.constrain_crf,
                        args.kingdom_as_token, args.global_label_as_input,
                        args.model_architecture,
                        model_seq=model_seq,
                        model_prompt=model_prompt)


  # Collect results + header info to build output df
  partitions = list(range(args.n_partitions))

  # for result collection
  metrics_list = []
  for partition in tqdm(partitions):
    # Load data
    dataset = ESM2CRFDataset(
        args.data,
        batch_converter=batch_converter,
        partition_id=[partition],
        add_special_tokens=True,
    )

    if args.randomize_kingdoms:
      kingdoms = list(set(dataset.kingdom_ids))
      random_kingdoms = random.choices(
          kingdoms, k=len(dataset.kingdom_ids))
      dataset.kingdom_ids = random_kingdoms
      print("randomized kingdom IDs")

    dl = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=50
    )
    
    
    model = PeftSPEsmCRF(config)
    if args.num_end_lora_layers > 0:
      print(f"Using LoRA: rank {args.num_lora_r}, "
            f"at last {args.num_end_lora_layers} layers.")
      target_modules = []
      start_layer_idx = num_esm_layers - args.num_end_lora_layers
      for idx in range(start_layer_idx, num_esm_layers):
        for layer_name in [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj","self_attn.out_proj"]:
          target_modules.append(f"layers.{idx}.{layer_name}")

      peft_config = LoraConfig(inference_mode=True,
                                r=args.num_lora_r,
                                lora_alpha=args.num_lora_alpha,
                                target_modules=target_modules,
                                lora_dropout=0.1,
                                bias="none",)
    model.load_state_dict(torch.load(checkpoint))


    # Put together list of checkpoints
    checkpoints = [
        os.path.join(
            args.model_base_path, f"test_{partition}_valid_{x}", "model.pt")
        for x in set(partitions).difference({partition})]

    # Run
    print(checkpoints)
    for checkpoint in checkpoints:
      model = PeftSPEsmCRF(config)

      if args.num_end_lora_layers > 0:
        print(f"Using LoRA: rank {args.num_lora_r}, "
              f"at last {args.num_end_lora_layers} layers.")
        target_modules = []
        start_layer_idx = num_esm_layers - args.num_end_lora_layers
        for idx in range(start_layer_idx, num_esm_layers):
          for layer_name in [
              "self_attn.q_proj", "self_attn.k_proj",
              "self_attn.v_proj","self_attn.out_proj"]:
            target_modules.append(f"layers.{idx}.{layer_name}")

        peft_config = LoraConfig(inference_mode=True,
                                 r=args.num_lora_r,
                                 lora_alpha=args.num_lora_alpha,
                                 target_modules=target_modules,
                                 lora_dropout=0.1,
                                 bias="none",)

        model = get_peft_model(model, peft_config)

      model.load_state_dict(torch.load(checkpoint))
      setattr(model, "use_pvd", args.use_pvd)  # ad hoc fix to use pvd

      metrics = get_metrics_multistate(
          model,
          dl,
          sp_tokens=[3, 7, 11, 15, 19] if args.no_multistate else None,
          compute_region_metrics=False, args=args
      )
      metrics_list.append(metrics)  # save metrics

  df = pd.DataFrame.from_dict(metrics_list)
  df.T.to_csv(args.output_file)


if __name__ == "__main__":
  main()
