### Part of the parameter description

- `--lr`, type=float, default=1e-4,
  peak learning rate.
- `--warmup_updates`, type=int, default=5000,
  learning rate warmup step.
- `--tot_updates`, type=int, default=350000,
  total steps of learning rate update.
- `--n_layers`, type=int, default=3,
  number of net layers.
- `--feature_dim`, type=int, default=384,
  dimensionality of hidden units in GNNs.
- `--head_num`, type=int, default=16,
  head number of Multi-head self attention.
- `--batch_size`, type=int, default=32,
  input batch size for training.
- `--infla`, type=float, default=[6.0],
  MCL encoding granularity can be changed by changing this parameter. Leaving this parameter unfilled means that the default INFLA(6) is used. This parameter can also be entered as an array of float types for a multi-scale encoding effect(For example: `--infla 3.0 6.0 9.0`).
- `--num_epochs`, type=int, default=150,
  number of epochs to train.
- `--atte_drop_rate`, type=float, default=0.5,
  attention dropout rate.
- `--input_drop_rate`, type=float, default=0.2,
  input dropout rate.
- `--layer_drop`, type=float, default=0.5,
  layer dropout rate. we use a variety of Dropout techniques (including query key value Drop, LayerDrop) mentioned in **UniDrop**<sup>[10]</sup>. 
- `--qkv_drop`, type=float, default=0.5,
  qkv dropout rate. we use a variety of Dropout techniques (including query key value Drop, LayerDrop) mentioned in **UniDrop**<sup>[10]</sup>. 



[1] Z. Wu *et al.*, “UniDrop: A Simple yet Effective Technique to Improve Transformer without Extra Cost,” *arXiv e-prints*, p. arXiv:2104.04946, Apr. 2021.