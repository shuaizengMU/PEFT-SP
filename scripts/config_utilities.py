
def load_basic_config(config, args):
  

  setattr(config, "num_labels", args.num_seq_labels)
  setattr(config, "num_global_labels", args.num_global_labels)

  setattr(config, "lm_output_dropout", args.lm_output_dropout)
  setattr(config, "lm_output_position_dropout", args.lm_output_position_dropout)
  setattr(config, "crf_scaling_factor", args.crf_scaling_factor)
  setattr(
      config, "use_large_crf", True
  )  # legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.

  if args.sp_region_labels:
      setattr(config, "use_region_labels", True)

  if args.kingdom_embed_size > 0:
      setattr(config, "use_kingdom_id", True)
      setattr(config, "kingdom_embed_size", args.kingdom_embed_size)

  # hardcoded for full model, 5 classes, 37 tags
  if args.constrain_crf and args.sp_region_labels:
      allowed_transitions = [
          # NO_SP
          (0, 0),
          (0, 1),
          (1, 1),
          (1, 2),
          (1, 0),
          (2, 1),
          (2, 2),  # I-I, I-M, M-M, M-O, M-I, O-M, O-O
          # SPI
          # 3 N, 4 H, 5 C, 6 I, 7M, 8 O
          (3, 3),
          (3, 4),
          (4, 4),
          (4, 5),
          (5, 5),
          (5, 8),
          (8, 8),
          (8, 7),
          (7, 7),
          (7, 6),
          (6, 6),
          (6, 7),
          (7, 8),
          # SPII
          # 9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
          (9, 9),
          (9, 10),
          (10, 10),
          (10, 11),
          (11, 11),
          (11, 12),
          (12, 15),
          (15, 15),
          (15, 14),
          (14, 14),
          (14, 13),
          (13, 13),
          (13, 14),
          (14, 15),
          # TAT
          # 16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
          (16, 16),
          (16, 17),
          (17, 17),
          (17, 16),
          (16, 18),
          (18, 18),
          (18, 19),
          (19, 19),
          (19, 22),
          (22, 22),
          (22, 21),
          (21, 21),
          (21, 20),
          (20, 20),
          (20, 21),
          (21, 22),
          # TATLIPO
          # 23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
          (23, 23),
          (23, 24),
          (24, 24),
          (24, 23),
          (23, 25),
          (25, 25),
          (25, 26),
          (26, 26),
          (26, 27),
          (27, 30),
          (30, 30),
          (30, 29),
          (29, 29),
          (29, 28),
          (28, 28),
          (28, 29),
          (29, 30),
          # PILIN
          # 31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
          (31, 31),
          (31, 32),
          (32, 32),
          (32, 33),
          (33, 33),
          (33, 36),
          (36, 36),
          (36, 35),
          (35, 35),
          (35, 34),
          (34, 34),
          (34, 35),
          (35, 36),
      ]
      #            'NO_SP_I' : 0,
      #            'NO_SP_M' : 1,
      #            'NO_SP_O' : 2,
      allowed_starts = [0, 2, 3, 9, 16, 23, 31]
      allowed_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]

      setattr(config, "allowed_crf_transitions", allowed_transitions)
      setattr(config, "allowed_crf_starts", allowed_starts)
      setattr(config, "allowed_crf_ends", allowed_ends)

  # setattr(config, 'gradient_checkpointing', True) #hardcoded when working with 256aa data
  if args.kingdom_as_token:
      setattr(
          config, "kingdom_id_as_token", True
      )  # model needs to know that token at pos 1 needs to be removed for CRF

  if args.global_label_as_input:
      setattr(config, "type_id_as_token", True)
      
  
  return config