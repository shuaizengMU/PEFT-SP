"""Utilities for ESM models.
"""
import esm_adapter

from . import prompts


def get_esm_pretained_model(model_architecture, num_end_adapter_layers):

  if model_architecture == "esm2_t48_15B_UR50D":
    return esm_adapter.pretrained.esm2_t48_15B_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t36_3B_UR50D":
    return esm_adapter.pretrained.esm2_t36_3B_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t33_650M_UR50D":
    return esm_adapter.pretrained.esm2_t33_650M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t30_150M_UR50D":
    return esm_adapter.pretrained.esm2_t30_150M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t12_35M_UR50D":
    return esm_adapter.pretrained.esm2_t12_35M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t6_8M_UR50D":
    return esm_adapter.pretrained.esm2_t6_8M_UR50D(num_end_adapter_layers)


def load_model(model_architecture, num_end_adapter_layers, freeze_backbone):

  esm2_model, _ = get_esm_pretained_model(model_architecture,
                                          num_end_adapter_layers)

  model_seq = prompts.EsmPromt(esm2_model, unfix_last_layer=0)

  print(sum(p.numel() for p in model_seq.parameters() if p.requires_grad))
  if freeze_backbone:
    for name, param in model_seq.named_parameters():
      if "adapter_layer" in name or "embeding_coder" in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
  print(sum(p.numel() for p in model_seq.parameters() if p.requires_grad))

  return model_seq
