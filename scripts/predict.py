"""Predict signal peptide type and cleavage site.
"""
import torch
import pandas as pd
import numpy as np

from peft import LoraConfig, get_peft_model
from collections import defaultdict

import args_maker
from signalp6.models import PeftSPEsmCRF
from signalp6.training_utils import ESM2CRFDPredictionataset
from signalp6.utils import tagged_seq_to_cs_multiclass
from signalp6.utils import metrics_utils
from model_config import PeftSPConfig
from utilities import esm_utilities, prompts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.hub.set_dir("/home/zengs/zengs_data/torch_hub")


def main():

  args = args_maker.prediction_args()
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

  # Load data
  dataset = ESM2CRFDPredictionataset(
      args.data,
      batch_converter=batch_converter,
  )

  dataloader = torch.utils.data.DataLoader(
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
              "self_attn.v_proj", "self_attn.out_proj"]:
        target_modules.append(f"layers.{idx}.{layer_name}")

    peft_config = LoraConfig(inference_mode=True,
                             r=args.num_lora_r,
                             lora_alpha=args.num_lora_alpha,
                             target_modules=target_modules,
                             lora_dropout=0.1,
                             bias="none",)
    model = get_peft_model(model, peft_config)

  model.load_state_dict(torch.load(args.model_filename))
  model.to(DEVICE)

  all_global_probs = []
  all_pos_preds = []
  for _, batch in enumerate(dataloader):
    (
        data,
        input_mask,
    ) = batch
    kingdom_ids = None

    data = data.to(DEVICE)
    input_mask = input_mask.to(DEVICE)
    with torch.no_grad():
      global_probs, _, pos_preds = model(
          data, input_mask=input_mask, kingdom_ids=kingdom_ids
      )

      all_global_probs.append(global_probs.detach().cpu().numpy())
      all_pos_preds.append(pos_preds.detach().cpu().numpy())

  all_global_probs = np.concatenate(all_global_probs)
  all_pos_preds = np.concatenate(all_pos_preds)

  all_global_preds = all_global_probs.argmax(axis=1)
  sp_tokens = [3, 7, 11, 15, 19]
  all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens=sp_tokens)

  # save results
  pred_dl = defaultdict(list)
  for idx in range(len(dataset)):
    sp_type_id = all_global_preds[idx]
    sp_type_name = metrics_utils.REVERSE_GLOBAL_LABEL_DICT[sp_type_id]
    cv_position = all_cs_preds[idx]
    sp_probs = np.round(all_global_probs[idx], 3)

    pred_dl["sp_type_id"].append(sp_type_id)
    pred_dl["sp_type_name"].append(sp_type_name)
    pred_dl["cv_position"].append(cv_position)
    pred_dl["sp_type_probabilities"].append(sp_probs)

  result_df = pd.DataFrame.from_dict(pred_dl)
  result_df.to_csv(args.output_file, index=False)
  
  print(result_df)


if __name__ == "__main__":
  main()
