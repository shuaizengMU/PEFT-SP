"""Utilities for adding prompts to ESM2 models.
"""
from esm_adapter.adapter import ResMLP

import torch
import torch.nn as nn


def from_sample_of_embeddings(embeddings, population_size=None):
  """Initialize by drawing vectors from the embedding table.
    
  Note:
    If not provided, the population size used is the full possibility of the
    vector space.
 
  Args:
    embeddings: [V, H] The embeddings to draw vectors from. can be extract 
      by model_seq.esm2.embed_tokens.weight
    population_size: Limit the drawing to the first `population_size` vectors.
  
  Returns:
    A closure over the embedding table that can be used as a flax initializer.
  """
  if population_size is None:
    population_size = embeddings.shape[0]
  if population_size <= 0:
    raise ValueError(f"Cannot sample from a population less than zero. Got "
                     f"{population_size}")
  if population_size > embeddings.shape[0]:
    # logging.warning(
    #    "The requested `population_size` (%d) is larger than the "
    #    "total available embeddings (%d). Setting "
    #    "`population_size` to the embedding size.", population_size,
    #    embeddings.shape[0])
    print("The requested `population_size` (%d) is larger than the "
          "total available embeddings (%d). Setting "
          "`population_size` to the embedding size.", population_size,
          embeddings.shape[0])

    population_size = embeddings.shape[0]

  # Because our sampling is done with jax (so that we can make use of the rng
  # key), we need our embeddings to be in jax, otherwise we get errors because
  # the indices will be a jax tracer and it fails when it is converted to numpy
  # to lookup values in a number array. This call pins the embeddings to cpu so
  # we don't waste TPU memory carrying it around.
  embeddings = embeddings.cpu()

  def initialize_from_embedding_sample(shape, rng=1234):
    """Sample from the embedding table, without replacement.
    
    Note:
      If the number of prompt tokens requested is larger than the total number
      of vectors we are drawing from (`population_size`) we do sampling with
      replacement.
    
    Args:
      rng: The rng seed used in our sampling.
      shape: The shape of the prompt variable. shape[0] tells us how many
        vectors to sample.
    
    Raises:
      ValueError if the number of features in the embedding table do not match
      the number of features in the prompt.
    
    Returns:
      A sample of the embedding table as a jax array. [P, H]
    """
    if embeddings.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    replace = False
    if shape[0] > population_size:
      print("Prompt Length: %d is larger than the number of vectors "
            "to draw from: %d. Switching to draws with replacement.", shape[0],
            population_size)
      replace = True

    # set the seed for torch random number generator
    torch.manual_seed(rng)
    if replace:
      index = torch.randint(population_size, size=(shape[0],))
    else:
      index = torch.multinomial(torch.ones(
          population_size), shape[0], replacement=False)

    return embeddings[index].clone().detach()

  return initialize_from_embedding_sample


def prefix_prompt(prompt, x_embed):
  """Concatenate `prompt` to the beginning of `x_embed`.
  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input.
    x: [B, T] The non-embedded input, used for finding the lengths of examples.
  
  Returns:
    The input with the prompt concatenated to the front. [B, P + T, H]
  """
  # del x
  return torch.cat((prompt, x_embed), dim=1)


def expand_to_batch(x, y):
  """Expand unbatched `x` to the same batch size as `y`."""
  batch_size = y.shape[0]
  expanded_x = x.unsqueeze(0)
  tiled_x = expanded_x.expand(batch_size, -1, -1)
  return tiled_x


class Prompts(nn.Module):
  """ A module that produces a learnable prompt.
  Example for defination: 
  prompt_initial_func=from_sample_of_embeddings
  x_embed_table=model_seq.esm2.embed_tokens.weight
  prompt = Prompts(args.prompt_len,prompt_initial_func,x_embed_table)
  """

  def __init__(self, length, prompt_initial_func, x_embed_table,
               num_esm_layers, original_seq_len=70,
               prompt_method=None, num_end_prompt_layers=0,
               res_mlp_bottleneck_size=0, device="cpu"):
    """_summary_

    Args:
        length (int): _description_
        prompt_initial_func (func): _description_
        x_embed_table (tensor): _description_
        num_esm_layers (int, optional): Number of layers in esm. 
          Defaults to None.
        prompt_layer_idx_list (List, optional): List of index of esm layer 
          indicating which layer adds prompt. The number must starts from 1. 
          Defaults to None.
    """
    super(Prompts, self).__init__()
    self.length = length
    self.original_seq_len = original_seq_len
    self.device = device

    embed_size = x_embed_table.shape[-1]

    # Prefix prompt
    # initialize prompt weight for each layer with None. The None values mean
    # the layer does not have prompt.
    self.prompt_weight_layers = [None for i in range(num_esm_layers)]

    if prompt_method == "SoftPromptFirst":
      prompt_layer_idx_list = [0]
    elif prompt_method == "SoftPromptAll":
      prompt_layer_idx_list = [i for i in range(0, num_esm_layers)]
    elif prompt_method == "SoftPromptLast":
      prompt_layer_idx_list = [num_esm_layers-1]
    elif prompt_method == "SoftPromptTopmost" and num_end_prompt_layers > 0:
      start_layer_idx = num_esm_layers - num_end_prompt_layers
      prompt_layer_idx_list = [
          idx for idx in range(start_layer_idx, num_esm_layers)]
    else:
      prompt_layer_idx_list = None

    if self.length > 0 and prompt_layer_idx_list is not None:

      # add prompt weight for specified layers.
      for idx in prompt_layer_idx_list:
        if idx == 0:
          initial_data = (prompt_initial_func(x_embed_table)(
              [self.length, embed_size]))
        else:
          initial_data = torch.randn(self.length, embed_size)
        self.prompt_weight_layers[idx] = nn.Parameter(initial_data)

        print(f"adding prompt to layer: {idx}")

    self.prompt_weight_layers = nn.ParameterList(self.prompt_weight_layers)
    self.combine = prefix_prompt

    self._initial_mlp_prompt_encoder(
        res_mlp_bottleneck_size, embed_size, prompt_layer_idx_list)

  def _initial_mlp_prompt_encoder(self, res_mlp_bottleneck_size,
                                  embed_size, prompt_layer_idx_list):

    if res_mlp_bottleneck_size > 0:
      # shared MLP encoder
      self.res_mlp = nn.ModuleDict({})
      for idx in prompt_layer_idx_list:
        self.res_mlp[f"{idx}"] = ResMLP(bottleneck_size=res_mlp_bottleneck_size,
                                        module_type="MLP1",
                                        dropout=0,
                                        emb_dimension=embed_size,
                                        nonlinearity="relu",
                                        layer_norm=True,
                                        residual=True,
                                        )
    else:
      self.res_mlp = None

  def forward(self, x_embed, layer_idx=0):
    if self.length == 0:
      return x_embed

    # Prompt weight is None taht represents no prompt for this layer.
    if self.prompt_weight_layers[layer_idx] is None:
      return x_embed

    # Getting output of sequence.
    # [B,prompt+T,E] => [B,T,E]
    if x_embed.shape[1] > self.original_seq_len+2:
      x_embed = x_embed[:, self.length:, :]

    # [B,T,E] => [B,prompt+T,E]
    prompt = expand_to_batch(
        self.prompt_weight_layers[layer_idx], x_embed)

    if self.res_mlp is not None and f"{layer_idx}" in self.res_mlp:
      prompt = self.res_mlp[f"{layer_idx}"](prompt)

    return self.combine(prompt, x_embed)


class EsmPromt(nn.Module):
  """ESM2 model with prompt.
  """

  def __init__(self, esm2_model, unfix_last_layer, *args, **kwargs):
    # Initialize the parent class using the parameters of esm2_model
    super(EsmPromt, self).__init__(*args, **kwargs)
    self.esm2 = esm2_model
    self.unfix_last_layer = unfix_last_layer

  def forward(self, model_prompt, tokens, repr_layers=None,
              need_head_weights=False, return_contacts=False):
    if return_contacts:
      need_head_weights = True
      
    if repr_layers is None:
      repr_layers = []

    # print("Device of input tokens:", tokens.device)
    assert tokens.ndim == 2

    prompt_len = model_prompt.length
    # if model_prompt.prompt_weight_layers[0] is not None:
    if model_prompt.prompt_weight_layers[0] is not None:

      # prefill_lengths = tokens.shape[1] + prompt_len
      # just give an unique token id to prompt tokens
      prefill_token_idx = len(self.esm2.alphabet.all_toks)+1
      prepand_tensor = torch.full(
          (tokens.shape[0], model_prompt.length),
          prefill_token_idx).to(tokens.device)
      # print("Device of input prepand_tensor:", prepand_tensor.device)
      # Concatenate the tensors and reshape
      prefill_tokens = torch.cat(
          (prepand_tensor, tokens), dim=1)  # B, prompt_length+T
      # print("Device of input prefill_tokens:", prefill_tokens.device)
      padding_mask = prefill_tokens.eq(
          self.esm2.padding_idx)  # B, (prompt_len+T)

    else:
      padding_mask = tokens.eq(self.esm2.padding_idx)  # B, T
      prefill_tokens = tokens

    # print("Device of input padding_mask:", padding_mask.device)
    x = self.esm2.embed_scale * self.esm2.embed_tokens(tokens)
    # input to TransformerLayer has been changed with prompt token.
    x = model_prompt(x, layer_idx=0)

    # print("Device of input x:", x.device)
    if self.esm2.token_dropout:
      # prompt token will dropout together with other nonpadding tokens.
      x.masked_fill_(
          (prefill_tokens == self.esm2.mask_idx).unsqueeze(-1), 0.0)
      # x: B x (prompt_len+T) x C
      mask_ratio_train = 0.15 * 0.8
      src_lengths = (~padding_mask).sum(-1)
      mask_ratio_observed = (
        (prefill_tokens == self.esm2.mask_idx).sum(-1).to(x.dtype)/src_lengths
        )
      x = x * (1 - mask_ratio_train) / \
          (1 - mask_ratio_observed)[:, None, None]

    if padding_mask is not None:
      x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
      hidden_representations[0] = x

    if need_head_weights:
      attn_weights = []

    # (B, prompt_len+T, E) => (prompt_len+T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
      padding_mask = None

    for layer_idx, layer in enumerate(self.esm2.layers):

      if (layer_idx > 0 and
          model_prompt.prompt_weight_layers[layer_idx] is not None):
        # (prompt_len+T, B, E) => (B, prompt_len+T, E)
        x = x.transpose(0, 1)

        # print("before prompt=======")
        # print(x, x.shape)
        # input to TransformerLayer has been changed with prompt token.
        x = model_prompt(x, layer_idx)

        # print("after prompt=====")
        # print(x, x.shape)
        # (B, prompt_len+T, E) => (prompt_len+T, B, E)
        x = x.transpose(0, 1)

        # normalizing after adding prompt tokens.
        # x = self.esm2.emb_layer_norm_after(x)

      x, attn = layer(
          x,
          # I didn't implement use self_attn_mask
          self_attn_padding_mask=padding_mask,
          need_head_weights=need_head_weights,
      )
      if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x.transpose(0, 1)
      if need_head_weights:
        # (H, B, prompt_len+T, prompt_len+T) =>
        # (B, H, prompt_len+T, prompt_len+T)
        attn_weights.append(attn.transpose(1, 0))

    x = self.esm2.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (prompt_len+T, B, E) => (B, prompt_len+T, E)
    # remove the prompt_tokens
    x = x[:, prompt_len:, :]  # => [B,T,E]
    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
      hidden_representations[layer_idx + 1] = x
    x = self.esm2.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
      # attentions: B x L x H x (prompt_len+T) x (prompt_len+T)
      attentions = torch.stack(attn_weights, 1)
      if padding_mask is not None:
        attention_mask = 1 - padding_mask.type_as(attentions)
        attention_mask = attention_mask.unsqueeze(
            1) * attention_mask.unsqueeze(2)
        attentions = attentions * attention_mask[:, None, None, :, :]

      result["attentions"] = attentions[:, :, :, prompt_len:, prompt_len:]
      if return_contacts:
        contacts = self.esm2.contact_head(
            tokens, attentions[:, :, :, prompt_len:, prompt_len:])
        result["contacts"] = contacts

    return result

  def predict_contacts(self, tokens):
    return self(tokens, return_contacts=True)["contacts"]
