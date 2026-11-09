nohup: ignoring input
Using device: cuda
✅ 已加载 ESM 特征: 165 条 | MolFormer 特征: 23 条 | MolFormer 特征: 12 条
Loaded df: (17470, 8)
📂 Enzyme-group split: train=12257, val=2607, test=2606
🧬 Unique enzymes: train=125, val=20, test=20
train: n=12257, active=2446, inactive=9811, active_ratio=0.1996
val: n=2607, active=514, inactive=2093, active_ratio=0.1972
test: n=2606, active=534, inactive=2072, active_ratio=0.2049
Positive weight (sqrt ratio): 2.002757706286495
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 1/200 | Train Loss=0.9680 (cls=0.6580, reg=0.2119, rank=0.6542) | Val Loss=0.7264 | F1=0.6613 | PR-AUC=0.6249 | Active Spearman=0.5737 | Top100 NDCG=0.5180 | Selection=0.6120 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 2/200 | Train Loss=0.7633 (cls=0.4939, reg=0.1771, rank=0.6156) | Val Loss=0.6879 | F1=0.6689 | PR-AUC=0.6759 | Active Spearman=0.6023 | Top100 NDCG=0.4626 | Selection=0.6006 | LR=1.00e-04
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 3/200 | Train Loss=0.7131 (cls=0.4622, reg=0.1603, rank=0.6043) | Val Loss=0.6494 | F1=0.6667 | PR-AUC=0.6798 | Active Spearman=0.5952 | Top100 NDCG=0.4959 | Selection=0.6173 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 4/200 | Train Loss=0.6821 (cls=0.4429, reg=0.1502, rank=0.5931) | Val Loss=0.6464 | F1=0.6755 | PR-AUC=0.7030 | Active Spearman=0.6085 | Top100 NDCG=0.5005 | Selection=0.6271 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 5/200 | Train Loss=0.6646 (cls=0.4338, reg=0.1435, rank=0.5819) | Val Loss=0.6408 | F1=0.6647 | PR-AUC=0.7230 | Active Spearman=0.6006 | Top100 NDCG=0.5056 | Selection=0.6336 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 6/200 | Train Loss=0.6437 (cls=0.4261, reg=0.1312, rank=0.5757) | Val Loss=0.6303 | F1=0.6783 | PR-AUC=0.7191 | Active Spearman=0.6081 | Top100 NDCG=0.5405 | Selection=0.6510 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 7/200 | Train Loss=0.6315 (cls=0.4176, reg=0.1286, rank=0.5689) | Val Loss=0.6732 | F1=0.6686 | PR-AUC=0.7364 | Active Spearman=0.6070 | Top100 NDCG=0.5307 | Selection=0.6503 | LR=1.00e-04
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 8/200 | Train Loss=0.6232 (cls=0.4125, reg=0.1260, rank=0.5643) | Val Loss=0.6323 | F1=0.6757 | PR-AUC=0.7417 | Active Spearman=0.6132 | Top100 NDCG=0.5396 | Selection=0.6569 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 9/200 | Train Loss=0.6128 (cls=0.4070, reg=0.1217, rank=0.5598) | Val Loss=0.6145 | F1=0.6742 | PR-AUC=0.7426 | Active Spearman=0.6086 | Top100 NDCG=0.5463 | Selection=0.6599 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 10/200 | Train Loss=0.5963 (cls=0.3991, reg=0.1144, rank=0.5520) | Val Loss=0.6314 | F1=0.7098 | PR-AUC=0.7443 | Active Spearman=0.5999 | Top100 NDCG=0.5527 | Selection=0.6624 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 11/200 | Train Loss=0.5925 (cls=0.3990, reg=0.1103, rank=0.5544) | Val Loss=0.6414 | F1=0.7039 | PR-AUC=0.7380 | Active Spearman=0.6111 | Top100 NDCG=0.5625 | Selection=0.6671 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 12/200 | Train Loss=0.5890 (cls=0.3942, reg=0.1117, rank=0.5537) | Val Loss=0.6170 | F1=0.7032 | PR-AUC=0.7465 | Active Spearman=0.6153 | Top100 NDCG=0.5598 | Selection=0.6684 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 13/200 | Train Loss=0.5712 (cls=0.3833, reg=0.1059, rank=0.5465) | Val Loss=0.6138 | F1=0.7115 | PR-AUC=0.7431 | Active Spearman=0.6157 | Top100 NDCG=0.5545 | Selection=0.6650 | LR=1.00e-04
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 14/200 | Train Loss=0.5756 (cls=0.3816, reg=0.1117, rank=0.5490) | Val Loss=0.6081 | F1=0.7106 | PR-AUC=0.7634 | Active Spearman=0.6093 | Top100 NDCG=0.5652 | Selection=0.6746 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 15/200 | Train Loss=0.5675 (cls=0.3816, reg=0.1043, rank=0.5433) | Val Loss=0.6139 | F1=0.7172 | PR-AUC=0.7624 | Active Spearman=0.6075 | Top100 NDCG=0.5679 | Selection=0.6755 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 16/200 | Train Loss=0.5588 (cls=0.3735, reg=0.1034, rank=0.5457) | Val Loss=0.6313 | F1=0.7131 | PR-AUC=0.7701 | Active Spearman=0.5982 | Top100 NDCG=0.5665 | Selection=0.6756 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 17/200 | Train Loss=0.5494 (cls=0.3654, reg=0.1028, rank=0.5410) | Val Loss=0.6086 | F1=0.7226 | PR-AUC=0.7576 | Active Spearman=0.6138 | Top100 NDCG=0.5555 | Selection=0.6689 | LR=1.00e-04
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 18/200 | Train Loss=0.5413 (cls=0.3624, reg=0.0983, rank=0.5379) | Val Loss=0.6266 | F1=0.6871 | PR-AUC=0.7784 | Active Spearman=0.6029 | Top100 NDCG=0.5902 | Selection=0.6900 | LR=1.00e-04
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 19/200 | Train Loss=0.5436 (cls=0.3610, reg=0.1011, rank=0.5434) | Val Loss=0.6135 | F1=0.7094 | PR-AUC=0.7791 | Active Spearman=0.6119 | Top100 NDCG=0.5668 | Selection=0.6797 | LR=1.00e-04
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 20/200 | Train Loss=0.5285 (cls=0.3477, reg=0.1003, rank=0.5371) | Val Loss=0.6413 | F1=0.6925 | PR-AUC=0.7703 | Active Spearman=0.5921 | Top100 NDCG=0.5801 | Selection=0.6817 | LR=5.00e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 21/200 | Train Loss=0.5129 (cls=0.3409, reg=0.0919, rank=0.5344) | Val Loss=0.6631 | F1=0.6802 | PR-AUC=0.7724 | Active Spearman=0.5907 | Top100 NDCG=0.5630 | Selection=0.6734 | LR=5.00e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 22/200 | Train Loss=0.5069 (cls=0.3346, reg=0.0924, rank=0.5318) | Val Loss=0.6550 | F1=0.6936 | PR-AUC=0.7777 | Active Spearman=0.5922 | Top100 NDCG=0.5809 | Selection=0.6839 | LR=5.00e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 23/200 | Train Loss=0.5093 (cls=0.3395, reg=0.0898, rank=0.5329) | Val Loss=0.6225 | F1=0.7256 | PR-AUC=0.7793 | Active Spearman=0.6126 | Top100 NDCG=0.5918 | Selection=0.6923 | LR=5.00e-05
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 24/200 | Train Loss=0.5030 (cls=0.3326, reg=0.0909, rank=0.5301) | Val Loss=0.6325 | F1=0.7053 | PR-AUC=0.7738 | Active Spearman=0.6006 | Top100 NDCG=0.5766 | Selection=0.6818 | LR=5.00e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 25/200 | Train Loss=0.4957 (cls=0.3246, reg=0.0917, rank=0.5295) | Val Loss=0.6567 | F1=0.6914 | PR-AUC=0.7775 | Active Spearman=0.5823 | Top100 NDCG=0.5599 | Selection=0.6721 | LR=5.00e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 26/200 | Train Loss=0.4982 (cls=0.3296, reg=0.0890, rank=0.5309) | Val Loss=0.6294 | F1=0.7087 | PR-AUC=0.7834 | Active Spearman=0.6044 | Top100 NDCG=0.5850 | Selection=0.6889 | LR=2.50e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 27/200 | Train Loss=0.4780 (cls=0.3154, reg=0.0838, rank=0.5248) | Val Loss=0.6204 | F1=0.7105 | PR-AUC=0.7834 | Active Spearman=0.6119 | Top100 NDCG=0.5856 | Selection=0.6901 | LR=2.50e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 28/200 | Train Loss=0.4842 (cls=0.3157, reg=0.0891, rank=0.5289) | Val Loss=0.6231 | F1=0.7157 | PR-AUC=0.7836 | Active Spearman=0.6063 | Top100 NDCG=0.5798 | Selection=0.6866 | LR=2.50e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 29/200 | Train Loss=0.4773 (cls=0.3123, reg=0.0860, rank=0.5266) | Val Loss=0.6304 | F1=0.7045 | PR-AUC=0.7825 | Active Spearman=0.6171 | Top100 NDCG=0.5900 | Selection=0.6928 | LR=2.50e-05
保存当前最佳模型（综合 Top100 / 分类 / 回归验证指标）！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 30/200 | Train Loss=0.4770 (cls=0.3108, reg=0.0871, rank=0.5274) | Val Loss=0.6422 | F1=0.7089 | PR-AUC=0.7767 | Active Spearman=0.5933 | Top100 NDCG=0.5681 | Selection=0.6774 | LR=2.50e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 31/200 | Train Loss=0.4741 (cls=0.3098, reg=0.0853, rank=0.5261) | Val Loss=0.6385 | F1=0.7112 | PR-AUC=0.7786 | Active Spearman=0.5929 | Top100 NDCG=0.5532 | Selection=0.6704 | LR=2.50e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 32/200 | Train Loss=0.4672 (cls=0.3042, reg=0.0845, rank=0.5234) | Val Loss=0.6395 | F1=0.6965 | PR-AUC=0.7778 | Active Spearman=0.5913 | Top100 NDCG=0.5593 | Selection=0.6730 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 33/200 | Train Loss=0.4694 (cls=0.3054, reg=0.0852, rank=0.5253) | Val Loss=0.6388 | F1=0.7049 | PR-AUC=0.7809 | Active Spearman=0.5973 | Top100 NDCG=0.5715 | Selection=0.6807 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 34/200 | Train Loss=0.4685 (cls=0.3066, reg=0.0829, rank=0.5268) | Val Loss=0.6381 | F1=0.7107 | PR-AUC=0.7817 | Active Spearman=0.6035 | Top100 NDCG=0.5924 | Selection=0.6921 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 35/200 | Train Loss=0.4669 (cls=0.3043, reg=0.0836, rank=0.5267) | Val Loss=0.6406 | F1=0.7097 | PR-AUC=0.7800 | Active Spearman=0.5982 | Top100 NDCG=0.5844 | Selection=0.6870 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 36/200 | Train Loss=0.4671 (cls=0.3024, reg=0.0858, rank=0.5258) | Val Loss=0.6408 | F1=0.7110 | PR-AUC=0.7821 | Active Spearman=0.5999 | Top100 NDCG=0.5744 | Selection=0.6827 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 37/200 | Train Loss=0.4641 (cls=0.3035, reg=0.0823, rank=0.5219) | Val Loss=0.6395 | F1=0.7059 | PR-AUC=0.7802 | Active Spearman=0.5983 | Top100 NDCG=0.5724 | Selection=0.6811 | LR=1.25e-05
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 38/200 | Train Loss=0.4671 (cls=0.3032, reg=0.0849, rank=0.5263) | Val Loss=0.6442 | F1=0.7121 | PR-AUC=0.7805 | Active Spearman=0.5923 | Top100 NDCG=0.5616 | Selection=0.6750 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 39/200 | Train Loss=0.4635 (cls=0.2988, reg=0.0858, rank=0.5261) | Val Loss=0.6470 | F1=0.7110 | PR-AUC=0.7811 | Active Spearman=0.5898 | Top100 NDCG=0.5622 | Selection=0.6751 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 40/200 | Train Loss=0.4620 (cls=0.2981, reg=0.0850, rank=0.5256) | Val Loss=0.6461 | F1=0.7090 | PR-AUC=0.7809 | Active Spearman=0.5910 | Top100 NDCG=0.5557 | Selection=0.6719 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 41/200 | Train Loss=0.4587 (cls=0.2953, reg=0.0848, rank=0.5243) | Val Loss=0.6407 | F1=0.7111 | PR-AUC=0.7821 | Active Spearman=0.5955 | Top100 NDCG=0.5647 | Selection=0.6773 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 42/200 | Train Loss=0.4517 (cls=0.2916, reg=0.0818, rank=0.5222) | Val Loss=0.6444 | F1=0.7111 | PR-AUC=0.7820 | Active Spearman=0.5964 | Top100 NDCG=0.5636 | Selection=0.6768 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 43/200 | Train Loss=0.4503 (cls=0.2911, reg=0.0808, rank=0.5225) | Val Loss=0.6466 | F1=0.7111 | PR-AUC=0.7847 | Active Spearman=0.5986 | Top100 NDCG=0.5797 | Selection=0.6859 | LR=6.25e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 44/200 | Train Loss=0.4573 (cls=0.2957, reg=0.0831, rank=0.5236) | Val Loss=0.6428 | F1=0.7124 | PR-AUC=0.7834 | Active Spearman=0.5962 | Top100 NDCG=0.5665 | Selection=0.6786 | LR=3.13e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 45/200 | Train Loss=0.4506 (cls=0.2913, reg=0.0809, rank=0.5223) | Val Loss=0.6446 | F1=0.7112 | PR-AUC=0.7830 | Active Spearman=0.5956 | Top100 NDCG=0.5667 | Selection=0.6785 | LR=3.13e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 46/200 | Train Loss=0.4537 (cls=0.2953, reg=0.0806, rank=0.5190) | Val Loss=0.6459 | F1=0.7089 | PR-AUC=0.7836 | Active Spearman=0.5951 | Top100 NDCG=0.5669 | Selection=0.6787 | LR=3.13e-06
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 47/200 | Train Loss=0.4528 (cls=0.2935, reg=0.0809, rank=0.5228) | Val Loss=0.6459 | F1=0.7105 | PR-AUC=0.7839 | Active Spearman=0.5946 | Top100 NDCG=0.5674 | Selection=0.6790 | LR=3.13e-06
Early Stopping at Epoch 47

训练结束，最佳验证综合分数：0.692778
训练期间最低验证损失：0.608140
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(

========== TEST RESULTS ==========
Accuracy: 0.861857252494244
Precision: 0.6385350318471338
Recall: 0.7509363295880149
F1: 0.6901893287435457
ROC_AUC: 0.9132738275996704
PR_AUC: 0.7263355203515963
MAE: 0.3066825594800218
RMSE: 1.10171138667259
R2: 0.2220783302247754
Active_MAE: 1.1259860687693826
Active_RMSE: 2.2898076008588757
Active_R2: 0.15609914476876363
Active_Spearman: 0.612239829945451
TopK: 100
TopK_Precision: 0.43
TopK_Recall: 0.43
TopK_NDCG: 0.5864920258173266
TopK_Mean_True_Activity: 3.0341940228641033
TopK_Mean_Predicted_Score: 4.684318107904612
TopK_Active_Ratio: 0.8
TopK_True_Active_Count: 534
Loss: 0.6893604803426111
测试集预测结果已保存：/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/model_activity_Activity_top/test_predictions.csv
测试集预测 Top100 已保存：/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/model_activity_Activity_top/test_top100_predictions.csv
