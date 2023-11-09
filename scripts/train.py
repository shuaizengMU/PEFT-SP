"""Main function to training PEFT-SP using PEFT method and ESM-2 models.
"""
import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np

from typing import Tuple, Dict
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from sklearn.metrics import (
    matthews_corrcoef,
    recall_score,
    precision_score,
)

import args_maker

from model_config import PeftSPConfig
from utilities import esm_utilities, prompts
from signalp6.models import PeftSPEsmCRF
from signalp6.training_utils import (
    SIGNALP_KINGDOM_DICT,
    ESM2CRFDataset,
    compute_cosine_region_regularization,
)
from signalp6.utils import get_metrics_multistate
from signalp6.utils import class_aware_cosine_similarities, get_region_lengths


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.hub.set_dir("./torch_hub")
torch.hub.set_dir("/home/zengs/zengs_data/torch_hub")


def setup_logger(output_dir: str = None):
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  c_handler = logging.StreamHandler()
  formatter = logging.Formatter(
      "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%y/%m/%d %H:%M:%S",
  )
  c_handler.setFormatter(formatter)
  logger.addHandler(c_handler)

  f_handler = logging.FileHandler(
      os.path.join(output_dir, "log.txt"))
  formatter = logging.Formatter(
      "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%y/%m/%d %H:%M:%S",
  )
  f_handler.setFormatter(formatter)
  logger.addHandler(f_handler)
  return logger


def set_randomseed(seed: int = None):
  if seed is not None:
    torch.manual_seed(seed)
    return seed
  else:
    return torch.seed()


def print_trainable_parameters(model):
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()

  return (f"trainable params: {trainable_params} || all params: {all_param} "
          f"|| trainable: {(100 * trainable_params / all_param)}")


def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens=None):
  """Convert a sequences of tokens to the index of the cleavage site.
  Inputs:
      tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
      sp_tokens: label tokens that indicate a signal peptide
  Returns:
      cs_sites: (batch_size) integer array of last position that is a SP. 
        NaN if no SP present in sequence.
  """
  if sp_tokens is None:
    sp_tokens = [0, 4, 5]

  def get_last_sp_idx(x: np.ndarray) -> int:
    """Func1d to get the last index that is tagged as SP. use with 
      np.apply_along_axis. 
    """
    sp_idx = np.where(np.isin(x, sp_tokens))[0]
    if len(sp_idx) > 0:
      max_idx = sp_idx.max() + 1
    else:
      max_idx = np.nan
    return max_idx

  cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
  return cs_sites


def report_metrics(
        true_global_labels: np.ndarray,
        pred_global_labels: np.ndarray,
        true_sequence_labels: np.ndarray,
        pred_sequence_labels: np.ndarray,
        use_cs_tag=False,) -> Dict[str, float]:
  """Utility function to get metrics from model output"""
  true_cs = tagged_seq_to_cs_multiclass(
      true_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
  )
  pred_cs = tagged_seq_to_cs_multiclass(
      pred_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
  )
  pred_cs = pred_cs[~np.isnan(true_cs)]
  true_cs = true_cs[~np.isnan(true_cs)]
  true_cs[np.isnan(true_cs)] = -1
  pred_cs[np.isnan(pred_cs)] = -1

  # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent
  # to the argmax.
  pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
  metrics_dict = {}
  metrics_dict["CS Recall"] = recall_score(true_cs, pred_cs, average="micro")
  metrics_dict["CS Precision"] = precision_score(
      true_cs, pred_cs, average="micro")
  metrics_dict["CS MCC"] = matthews_corrcoef(true_cs, pred_cs)
  metrics_dict["Detection MCC"] = matthews_corrcoef(
      true_global_labels, pred_global_labels_thresholded
  )

  return metrics_dict


def report_metrics_kingdom_averaged(
        true_global_labels: np.ndarray,
        pred_global_labels: np.ndarray,
        true_sequence_labels: np.ndarray,
        pred_sequence_labels: np.ndarray,
        kingdom_ids: np.ndarray,
        input_token_ids: np.ndarray,
        cleavage_sites: np.ndarray = None,
        use_cs_tag=False,
        args=None,) -> Dict[str, float]:
  """Utility function to get metrics from model output"""

  sp_tokens = [3, 7, 11, 15, 19]
  if use_cs_tag:
    sp_tokens = [4, 9, 14]

  # implicit: when cleavage sites are provided, am using region states
  if cleavage_sites is not None:
    sp_tokens = [5, 11, 19, 26, 31]
    true_cs = cleavage_sites.astype(float)
    # need to convert so np.isnan works
    true_cs[true_cs == -1] = np.nan
  else:
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels,
                                          sp_tokens=sp_tokens)

  pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels,
                                        sp_tokens=sp_tokens)

  cs_kingdom = kingdom_ids[~np.isnan(true_cs)]
  pred_cs = pred_cs[~np.isnan(true_cs)]
  true_cs = true_cs[~np.isnan(true_cs)]
  true_cs[np.isnan(true_cs)] = -1
  pred_cs[np.isnan(pred_cs)] = -1

  # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent
  # to the argmax.
  pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

  # compute metrics for each kingdom
  rev_kingdom_dict = dict(
      zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys())
  )
  all_cs_mcc = []
  all_detection_mcc = []
  metrics_dict = {}
  for kingdom in np.unique(kingdom_ids):
    kingdom_global_labels = true_global_labels[kingdom_ids == kingdom]
    kingdom_pred_global_labels_thresholded = pred_global_labels_thresholded[
        kingdom_ids == kingdom
    ]
    kingdom_true_cs = true_cs[cs_kingdom == kingdom]
    kingdom_pred_cs = pred_cs[cs_kingdom == kingdom]

    metrics_dict[f"CS Recall {rev_kingdom_dict[kingdom]}"] = recall_score(
        kingdom_true_cs, kingdom_pred_cs, average="micro"
    )
    metrics_dict[f"CS Precision {rev_kingdom_dict[kingdom]}"] = precision_score(
        kingdom_true_cs, kingdom_pred_cs, average="micro"
    )
    metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"] = matthews_corrcoef(
        kingdom_true_cs, kingdom_pred_cs
    )
    metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"] = (
        matthews_corrcoef(kingdom_global_labels,
                          kingdom_pred_global_labels_thresholded)
    )

    all_cs_mcc.append(metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"])
    all_detection_mcc.append(
        metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"]
    )

  # implicit: when cleavage sites are provided, am using region states
  if cleavage_sites is not None:
    n_h, h_c = class_aware_cosine_similarities(
        pred_sequence_labels,
        input_token_ids,
        true_global_labels,
        replace_value=np.nan,
        op_mode="numpy",
        args=args,
    )
    n_lengths, h_lengths, c_lengths = get_region_lengths(
        pred_sequence_labels, true_global_labels, agg_fn="none"
    )
    for label in np.unique(true_global_labels):
      if label == 0 or label == 5:
        continue

      metrics_dict[f"Cosine similarity nh {label}"] = np.nanmean(
          n_h[true_global_labels == label]
      )
      metrics_dict[f"Cosine similarity hc {label}"] = np.nanmean(
          h_c[true_global_labels == label]
      )
      metrics_dict[f"Average length n {label}"] = n_lengths[
          true_global_labels == label
      ].mean()
      metrics_dict[f"Average length h {label}"] = h_lengths[
          true_global_labels == label
      ].mean()
      metrics_dict[f"Average length c {label}"] = c_lengths[
          true_global_labels == label
      ].mean()
      # w&b can plot histogram heatmaps over time when logging sequences
      metrics_dict[f"Lengths n {label}"] = (
          n_lengths[true_global_labels == label])
      metrics_dict[f"Lengths h {label}"] = (
          h_lengths[true_global_labels == label])
      metrics_dict[f"Lengths c {label}"] = (
          c_lengths[true_global_labels == label])

  metrics_dict["CS MCC"] = sum(all_cs_mcc) / len(all_cs_mcc)
  metrics_dict["Detection MCC"] = (
      sum(all_detection_mcc) / len(all_detection_mcc))

  return metrics_dict


def train_model(
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        optimizer_classifier: torch.optim.Optimizer,
        args: argparse.ArgumentParser,
        global_step: int,) -> Tuple[float, int]:
  """Predict one minibatch and performs update step.
  Returns:
      loss: loss value of the minibatch
  """

  first_seq_idx = 1
  if args.kingdom_as_token:
    first_seq_idx = 2

  optimizer.zero_grad()

  all_targets = []
  all_global_targets = []
  all_global_probs = []
  all_pos_preds = []
  all_kingdom_ids = []  # gather ids for kingdom-averaged metrics
  all_token_ids = []
  all_cs = []
  total_loss = 0
  for _, batch in enumerate(train_data):
    if args.sp_region_labels:
      (
          data,
          targets,
          input_mask,
          global_targets,
          cleavage_sites,
          sample_weights,
          kingdom_ids,
      ) = batch
    else:
      (
          data,
          targets,
          input_mask,
          global_targets,
          sample_weights,
          kingdom_ids,
      ) = batch

    data = data.to(DEVICE)
    targets = targets.to(DEVICE)
    input_mask = input_mask.to(DEVICE)
    global_targets = global_targets.to(DEVICE)
    sample_weights = sample_weights.to(
        DEVICE) if args.use_sample_weights else None
    kingdom_ids = kingdom_ids.to(DEVICE)

    optimizer.zero_grad()

    if optimizer_classifier is not None:
      optimizer_classifier.zero_grad()

    loss, global_probs, pos_probs, pos_preds = model(
        data,
        global_targets=None,
        targets=targets if not args.sp_region_labels else None,
        targets_bitmap=targets if args.sp_region_labels else None,
        input_mask=input_mask,
        sample_weights=sample_weights,
        kingdom_ids=kingdom_ids if args.kingdom_embed_size > 0 else None,
    )
    loss = (
        loss.mean()
    )  # if DataParallel because loss is a vector, if not doesn't matter

    total_loss += loss.item()
    all_targets.append(targets.detach().cpu().numpy())
    all_global_targets.append(global_targets.detach().cpu().numpy())
    all_global_probs.append(global_probs.detach().cpu().numpy())
    all_pos_preds.append(pos_preds.detach().cpu().numpy())
    all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
    all_token_ids.append(data.detach().cpu().numpy())
    all_cs.append(cleavage_sites if args.sp_region_labels else None)

    # print(loss, global_probs, pos_probs, pos_preds)
    # exit(0)

    # if args.region_regularization_alpha >0:
    # removing special tokens by indexing should be sufficient.
    # remaining SEP tokens (when sequence was padded)
    # are ignored in aggregation.
    if args.sp_region_labels and args.region_regularization_alpha > 0:
      nh, hc = compute_cosine_region_regularization(
          pos_probs, data[:, first_seq_idx:-1], global_targets,
          input_mask[:, first_seq_idx:-1]
      )
      loss = loss + nh.mean() * args.region_regularization_alpha
      loss = loss + hc.mean() * args.region_regularization_alpha

    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem
    # in RNNs / LSTMs.
    if args.clip:
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    # from IPython import embed; embed()
    optimizer.step()
    if optimizer_classifier is not None:
      optimizer_classifier.step()

    global_step += 1

  all_targets = np.concatenate(all_targets)
  all_global_targets = np.concatenate(all_global_targets)
  all_global_probs = np.concatenate(all_global_probs)
  all_pos_preds = np.concatenate(all_pos_preds)
  all_kingdom_ids = np.concatenate(all_kingdom_ids)
  all_token_ids = np.concatenate(all_token_ids)
  all_cs = np.concatenate(all_cs) if args.sp_region_labels else None

  if args.average_per_kingdom:
    _ = report_metrics_kingdom_averaged(
        all_global_targets,
        all_global_probs,
        all_targets,
        all_pos_preds,
        all_kingdom_ids,
        all_token_ids,
        all_cs,
        args.use_cs_tag,
        args=args,
    )
  else:
    _ = report_metrics(
        all_global_targets,
        all_global_probs,
        all_targets,
        all_pos_preds,
        args.use_cs_tag,
    )

  return total_loss / len(train_data), global_step


def validate_model(model: torch.nn.Module, valid_data: DataLoader,
                   args) -> float:
  """Run over the validation data. Average loss over the full set."""
  model.eval()

  all_targets = []
  all_global_targets = []
  all_global_probs = []
  all_pos_preds = []
  all_kingdom_ids = []
  all_token_ids = []
  all_cs = []

  total_loss = 0
  for _, batch in enumerate(valid_data):
    if args.sp_region_labels:
      (
          data,
          targets,
          input_mask,
          global_targets,
          cleavage_sites,
          sample_weights,
          kingdom_ids,
      ) = batch
    else:
      (
          data,
          targets,
          input_mask,
          global_targets,
          sample_weights,
          kingdom_ids,
      ) = batch
    data = data.to(DEVICE)
    targets = targets.to(DEVICE)
    input_mask = input_mask.to(DEVICE)
    global_targets = global_targets.to(DEVICE)
    sample_weights = sample_weights.to(
        DEVICE) if args.use_sample_weights else None
    kingdom_ids = kingdom_ids.to(DEVICE)

    with torch.no_grad():
      loss, global_probs, _, pos_preds = model(
          data,
          global_targets=None,
          targets=targets if not args.sp_region_labels else None,
          targets_bitmap=targets if args.sp_region_labels else None,
          sample_weights=sample_weights,
          input_mask=input_mask,
          kingdom_ids=kingdom_ids if args.kingdom_embed_size > 0 else None,
      )

    total_loss += loss.mean().item()
    all_targets.append(targets.detach().cpu().numpy())
    all_global_targets.append(global_targets.detach().cpu().numpy())
    all_global_probs.append(global_probs.detach().cpu().numpy())
    all_pos_preds.append(pos_preds.detach().cpu().numpy())
    all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
    all_token_ids.append(data.detach().cpu().numpy())
    all_cs.append(cleavage_sites if args.sp_region_labels else None)

  all_targets = np.concatenate(all_targets)
  all_global_targets = np.concatenate(all_global_targets)
  all_global_probs = np.concatenate(all_global_probs)
  all_pos_preds = np.concatenate(all_pos_preds)
  all_kingdom_ids = np.concatenate(all_kingdom_ids)
  all_token_ids = np.concatenate(all_token_ids)
  all_cs = np.concatenate(all_cs) if args.sp_region_labels else None

  if args.average_per_kingdom:
    metrics = report_metrics_kingdom_averaged(
        all_global_targets,
        all_global_probs,
        all_targets,
        all_pos_preds,
        all_kingdom_ids,
        all_token_ids,
        all_cs,
        args.use_cs_tag,
        args=args,
    )
  else:
    metrics = report_metrics(
        all_global_targets,
        all_global_probs,
        all_targets,
        all_pos_preds,
        args.use_cs_tag,
    )

  val_metrics = {"loss": total_loss / len(valid_data), **metrics}
  return (total_loss / len(valid_data)), val_metrics


def main(args):
  """main function to training model.

  Args:
      args (object): args from command line.
  """
  assert args.test_partition != args.validation_partition, (
      "test_partition and validation_partition must be different")

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

  full_name = "_".join(
      [
          "test",
          str(args.test_partition),
          "valid",
          str(args.validation_partition)
      ]
  )

  experiment_output_dir = os.path.join(
      args.output_dir, args.experiment_name, full_name)
  if not os.path.exists(experiment_output_dir):
    os.makedirs(experiment_output_dir, exist_ok=True)

  logger = setup_logger(experiment_output_dir)
  seed = set_randomseed(args.random_seed)
  # Set seed

  logger.info("args: %s", args)
  logger.info("torch seed: %s", seed)
  logger.info("Running model on %s, not using nvidia apex", DEVICE)
  logger.info("Saving to %s", experiment_output_dir)
  logger.info("Loading pretrained model in %s", args.resume)
  logger.info("Dataset: %s", args.data)
  logger.info("Learnign rate: %s", args.lr)
  logger.info("PEFT-SP Prompt method: %s", args.prompt_method)
  logger.info("PEFT-SP Prompt Length: %s", args.prompt_len)
  logger.info("PEFT-SP ResMLP bottleneck: %s", args.res_mlp_bottleneck_size)
  logger.info("PEFT-SP Number of prompts at the end of ESM-2 backbone: %s",
              args.num_end_prompt_layers)
  logger.info("PEFT-SP Backbone: %s", args.model_architecture)
  logger.info("PEFT-SP Number of adapters at the end of ESM-2 backbone: %s",
              args.num_end_adapter_layers)
  logger.info("PEFT-SP LoRA: rank %s, at last %s layers.",
              args.num_lora_r, args.num_end_lora_layers)

  # backbnone model
  model_seq = esm_utilities.load_model(args.model_architecture,
                                       args.num_end_adapter_layers,
                                       args.freeze_backbone).to(DEVICE)
  alphabet = model_seq.esm2.alphabet
  batch_converter = alphabet.get_batch_converter(
      truncation_seq_length=args.max_length)

  # prompt tuning model
  prompt_initial_func = prompts.from_sample_of_embeddings
  x_embed_table = model_seq.esm2.embed_tokens.weight
  num_esm_layers = len(model_seq.esm2.layers)
  model_prompt = prompts.Prompts(
      args.prompt_len, prompt_initial_func, x_embed_table,
      num_esm_layers=num_esm_layers,
      original_seq_len=70,
      prompt_method=args.prompt_method,
      num_end_prompt_layers=args.num_end_prompt_layers,
      res_mlp_bottleneck_size=args.res_mlp_bottleneck_size, device=DEVICE)

  # PEFT-SP using PEFT method and ESM-2 backbone.
  config = PeftSPConfig(args.num_seq_labels, args.num_global_labels,
                        args.lm_output_dropout, args.lm_output_position_dropout,
                        args.crf_scaling_factor, args.sp_region_labels,
                        args.kingdom_embed_size, args.constrain_crf,
                        args.kingdom_as_token, args.global_label_as_input,
                        args.model_architecture,
                        model_seq=model_seq,
                        model_prompt=model_prompt)

  model = PeftSPEsmCRF(config)

  # PEFT-SP using LoRA
  if args.num_end_lora_layers > 0:
    target_modules = []
    start_layer_idx = num_esm_layers - args.num_end_lora_layers
    for idx in range(start_layer_idx, num_esm_layers):
      for layer_name in ["self_attn.q_proj", "self_attn.k_proj",
                         "self_attn.v_proj", "self_attn.out_proj"]:
        target_modules.append(f"layers.{idx}.{layer_name}")

    peft_config = LoraConfig(inference_mode=False,
                             r=args.num_lora_r,
                             lora_alpha=args.num_lora_alpha,
                             target_modules=target_modules,
                             lora_dropout=0.1,
                             bias="none",)
    model = get_peft_model(model, peft_config)

  logger.info(print_trainable_parameters(model))
  logger.info(model)

  # # make CRF trainable.
  # for name, param in model.named_parameters():
  #   if ("outputs_to_emissions" in name or "crf.transitions" in name or
  #           "crf.start_transitions" in name or "crf.end_transitions" in name):
  #     param.requires_grad = True

  logger.info("**********************************")
  logger.info("Trainbel parameters:")
  for name, param in model.named_parameters():
    if param.requires_grad:
      logger.info("%s, %s", name, str(param.data.shape))
  logger.info("**********************************")

  # setup data
  val_id = args.validation_partition
  test_id = args.test_partition
  train_ids = [0, 1, 2]
  train_ids.remove(val_id)
  train_ids.remove(test_id)
  logger.info("Training on %s, validating on %d", str(train_ids), val_id)

  kingdoms = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"]
  if args.sp_region_labels:
    train_data = ESM2CRFDataset(
        args.data,
        args.sample_weights,
        batch_converter=batch_converter,
        partition_id=train_ids,
        kingdom_id=kingdoms,
        add_special_tokens=True,
        return_kingdom_ids=True,
        positive_samples_weight=args.positive_samples_weight,
        make_cs_state=args.use_cs_tag,
        add_global_label=args.global_label_as_input,
        augment_data_paths=[args.additional_train_set],
    )
    val_data = ESM2CRFDataset(
        args.data,
        args.sample_weights,
        batch_converter=batch_converter,
        partition_id=[val_id],
        kingdom_id=kingdoms,
        add_special_tokens=True,
        return_kingdom_ids=True,
        positive_samples_weight=args.positive_samples_weight,
        make_cs_state=args.use_cs_tag,
        add_global_label=args.global_label_as_input,
    )
    logger.info("Using labels for SP region prediction.")
  else:
    assert False, "Not implemented for non region labels"

  logger.info(
      "%d training sequences, %d validation sequences.",
      len(train_data), len(val_data)
  )

  train_loader = DataLoader(
      train_data,
      batch_size=args.batch_size,
      collate_fn=train_data.collate_fn,
      shuffle=True,
  )
  val_loader = DataLoader(
      val_data, batch_size=args.batch_size, collate_fn=train_data.collate_fn
  )

  logger.info("Data loaded. One epoch = %d batches.", len(train_loader))
  logger.info("Saving checkpoints at %s", experiment_output_dir)

  if args.num_end_lora_layers > 0:
    logger.info("Different optimizer for ESM and CRF.")

    transformer_to_optimize = (
      model.base_model.model.model_seq.esm2.layers.parameters())
    optimizer = torch.optim.Adamax(
        transformer_to_optimize, lr=args.lr, weight_decay=args.wdecay
    )
    optimizer_classifier = None
    # classifier_to_optimize = (
    #     list(model.base_model.model.outputs_to_emissions.parameters())
    #     + list(model.base_model.model.crf.parameters()))
    # optimizer_classifier = torch.optim.Adamax(
    #     classifier_to_optimize, lr=0.0000001, weight_decay=args.wdecay
    # )

  else:
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay
    )
    optimizer_classifier = None

  model.to(DEVICE)

  learning_rate_steps = 0
  num_epochs_no_improvement = 0
  global_step = 0
  best_mcc_sum = 0
  best_mcc_global = 0
  best_mcc_cs = 0
  for epoch in range(1, args.epochs + 1):
    logger.info("Starting epoch %d", epoch)

    epoch_loss, global_step = train_model(
      model, train_loader, optimizer, optimizer_classifier,
      args, global_step
    )
    logger.info("train loss: %f", epoch_loss)

    logger.info(
      "Step %d, Epoch %d: validating for %d Validation steps", global_step,
      epoch, len(val_loader)
    )

    val_loss, val_metrics = validate_model(model, val_loader, args)

    logger.info(
        "Validation: Loss %.4f, MCC global %.4f, MCC seq %.4f. "
        "Epochs without improvement: %d. Learning rate step: %d",
        val_loss, val_metrics["Detection MCC"], val_metrics["CS MCC"],
        num_epochs_no_improvement,learning_rate_steps
    )

    mcc_sum = val_metrics["Detection MCC"] + val_metrics["CS MCC"]
    if mcc_sum > best_mcc_sum:
      best_mcc_sum = mcc_sum
      best_mcc_global = val_metrics["Detection MCC"]
      best_mcc_cs = val_metrics["CS MCC"]
      num_epochs_no_improvement = 0

      # model.save_pretrained(experiment_output_dir)
      model_output = experiment_output_dir + "/model.pt"
      torch.save(model.state_dict(), model_output)
      logger.info(
          "New best model with loss %f,MCC Sum %f MCC global %f, "
          "MCC seq %f, Saving model, training step %d",
          val_loss, mcc_sum, val_metrics["Detection MCC"],
          val_metrics["CS MCC"], global_step
      )

    else:
      num_epochs_no_improvement += 1

      # when cross-validating, check that the seed is working
      # for region detection
      if args.crossval_run and epoch == 1:
        # small length in first epoch = bad seed.
        if val_metrics["Average length n 1"] <= 4:
          print("Bad seed for region tagging.")
          run_completed = False
          return best_mcc_global, best_mcc_cs, run_completed

  logger.info("Epoch %d, epoch limit reached. Training complete", epoch)
  logger.info(
      "Best: MCC Sum %f, Detection %f, CS %f",
      best_mcc_sum, best_mcc_global, best_mcc_cs
  )

  print_all_final_metrics = True
  if print_all_final_metrics == True:
    model = PeftSPEsmCRF(config)

    # Lora
    if args.num_end_lora_layers > 0:
      target_modules = []
      start_layer_idx = num_esm_layers - args.num_end_lora_layers
      for idx in range(start_layer_idx, num_esm_layers):
        for layer_name in ["self_attn.q_proj", "self_attn.k_proj",
                           "self_attn.v_proj", "self_attn.out_proj"]:
          target_modules.append(f"layers.{idx}.{layer_name}")

      peft_config = LoraConfig(inference_mode=True,
                               r=args.num_lora_r,
                               lora_alpha=args.num_lora_alpha,
                               target_modules=target_modules,
                               lora_dropout=0.1,
                               bias="none",)

      model = get_peft_model(model, peft_config)

    model_output = experiment_output_dir + "/model.pt"
    model.load_state_dict(torch.load(model_output))

    ds = ESM2CRFDataset(
        args.data,
        args.sample_weights,
        batch_converter=batch_converter,
        partition_id=[test_id],
        kingdom_id=kingdoms,
        add_special_tokens=True,
        return_kingdom_ids=True,
        positive_samples_weight=args.positive_samples_weight,
        make_cs_state=args.use_cs_tag,
        add_global_label=args.global_label_as_input,
    )
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=ds.collate_fn, batch_size=80
    )
    metrics = get_metrics_multistate(model, dataloader, args=args)
    val_metrics = get_metrics_multistate(model, val_loader, args=args)

    df = pd.DataFrame.from_dict([metrics, val_metrics]).T
    df.columns = ["test", "val"]
    df.index = df.index.str.split("_", expand=True)
    pd.set_option("display.max_rows", None)

    df.to_csv(os.path.join(experiment_output_dir, "metrics.csv"))

  return best_mcc_global


if __name__ == "__main__":
  args = args_maker.peft_sp_training_args()
  main(args)
