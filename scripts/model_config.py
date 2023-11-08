"""_summary_
"""

ESM_HIDDEN_SIZE_DICT = {
  "esm2_t48_15B_UR50D" : 5120,
  "esm2_t36_3B_UR50D" : 2560,
  "esm2_t33_650M_UR50D" : 1280,
  "esm2_t30_150M_UR50D" : 640,
  "esm2_t12_35M_UR50D" : 480,
}

class PeftSPConfig():
  """
  This is the configuration class to store the configuration of a model that
  consists of a ESM model and a CRF layer.
  """

  def __init__(self,
               num_labels: int,
               num_global_labels: int,
               lm_output_dropout: float,
               lm_output_position_dropout: float,
               crf_scaling_factor: float,
               sp_region_labels: bool,
               kingdom_embed_size: int,
               constrain_crf: bool,
               kingdom_as_token: bool,
               global_label_as_input: bool,
               model_architecture: str,
               model_seq: str,
               model_prompt: str,
               ):
    """Initialization for EsmCRFConfig.
    
    Args:
        num_labels (int): number of labels each position of sequence.
        num_global_labels (int): number of global labels.
        lm_output_dropout (float): dropout rate for the output of ESM.
        lm_output_position_dropout (float): dropout applied to LM output, 
            drops full hidden states from sequence.
        crf_scaling_factor (float): scaling factor for CRF loss.
        sp_region_labels (bool): Use Signal Peptide region labels or not.
        kingdom_embed_size (int): Size of kingdom embedding.
        constrain_crf (bool): Whether to constrain CRF transitions.
        kingdom_as_token (bool): Whether to use kingdom id as token.
        global_label_as_input (bool): Whether to use global label as input.
        model_architecture (int): The architecture of the ESM-2 model.
        model_seq (str): Model of backbone.
        model_prompt (str): Model of prompt.
    """

    self.num_labels = num_labels
    self.num_global_labels = num_global_labels
    self.lm_output_dropout = lm_output_dropout
    self.lm_output_position_dropout = lm_output_position_dropout
    self.crf_scaling_factor = crf_scaling_factor
    self.use_large_crf = True
    self.use_region_labels = sp_region_labels
    self.kingdom_id_as_token = kingdom_as_token
    self.type_id_as_token = global_label_as_input
    self.model_seq = model_seq
    self.model_prompt = model_prompt
    self.hidden_size = ESM_HIDDEN_SIZE_DICT[model_architecture]

    if kingdom_embed_size > 0:
      self.use_kingdom_id = True
      self.kingdom_embed_size = kingdom_as_token  

    if constrain_crf and sp_region_labels:
      allowed_transitions = [
          # NO_SP
          # I-I, I-M, M-M, M-O, M-I, O-M, O-O
          (0, 0), (0, 1), (1, 1), (1, 2), (1, 0), (2, 1), (2, 2),
          # SPI
          # 3 N, 4 H, 5 C, 6 I, 7M, 8 O
          (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 8), (8, 8), (8, 7),
          (7, 7), (7, 6), (6, 6), (6, 7), (7, 8),
          # SPII
          # 9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
          (9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12), (12, 15),
          (15, 15), (15, 14), (14, 14), (14, 13), (13, 13), (13, 14), (14, 15),
          # TAT
          # 16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
          (16, 16), (16, 17), (17, 17), (17, 16), (16, 18), (18, 18), (18, 19),
          (19, 19), (19, 22), (22, 22), (22, 21), (21, 21), (21, 20), (20, 20),
          (20, 21), (21, 22),
          # TATLIPO
          # 23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
          (23, 23), (23, 24), (24, 24), (24, 23), (23, 25), (25, 25), (25, 26),
          (26, 26), (26, 27), (27, 30), (30, 30), (30, 29), (29, 29), (29, 28),
          (28, 28), (28, 29), (29, 30),
          # PILIN
          # 31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
          (31, 31), (31, 32), (32, 32), (32, 33), (33, 33), (33, 36), (36, 36),
          (36, 35), (35, 35), (35, 34), (34, 34), (34, 35), (35, 36),
      ]
      allowed_starts = [0, 2, 3, 9, 16, 23, 31]
      allowed_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]

      self.allowed_crf_transitions = allowed_transitions
      self.allowed_crf_starts = allowed_starts
      self.allowed_crf_ends = allowed_ends
