nohup: 忽略输入
Device: cuda | Task: Pure Activity Regression
ESM features: 146 | MolFormer features: 102
Samples: {'train': 3761, 'val': 431, 'test': 606} | Enzymes: {'train': 116, 'val': 15, 'test': 15}
train: zero=144, mean=0.1635, std=1.1508, max=22.5160
val: zero=13, mean=0.2776, std=1.1994, max=14.3150
test: zero=16, mean=0.4636, std=2.6717, max=32.0800
预训练权重匹配比例: 1.0000，匹配层数: 65

========== STAGE 1: TRAIN FC4 ==========
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 001/50 | TrainLoss=0.361135 | ValLoss=0.480799 | ValRMSE=1.219704 | MAE=0.373177 | R2=-0.0365
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 002/50 | TrainLoss=0.296222 | ValLoss=0.468549 | ValRMSE=1.225049 | MAE=0.363344 | R2=-0.0456
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 003/50 | TrainLoss=0.265382 | ValLoss=0.445186 | ValRMSE=1.223218 | MAE=0.349663 | R2=-0.0425
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 004/50 | TrainLoss=0.240062 | ValLoss=0.422938 | ValRMSE=1.221209 | MAE=0.339300 | R2=-0.0391
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 005/50 | TrainLoss=0.223024 | ValLoss=0.406978 | ValRMSE=1.220517 | MAE=0.330847 | R2=-0.0379
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 006/50 | TrainLoss=0.211814 | ValLoss=0.399952 | ValRMSE=1.221173 | MAE=0.325567 | R2=-0.0390
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 007/50 | TrainLoss=0.200295 | ValLoss=0.384548 | ValRMSE=1.218593 | MAE=0.319102 | R2=-0.0346
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 008/50 | TrainLoss=0.193464 | ValLoss=0.372511 | ValRMSE=1.216129 | MAE=0.313972 | R2=-0.0304
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 009/50 | TrainLoss=0.188300 | ValLoss=0.363094 | ValRMSE=1.214902 | MAE=0.309543 | R2=-0.0283
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 010/50 | TrainLoss=0.184722 | ValLoss=0.357630 | ValRMSE=1.213860 | MAE=0.306060 | R2=-0.0266
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 011/50 | TrainLoss=0.181756 | ValLoss=0.351173 | ValRMSE=1.213431 | MAE=0.302683 | R2=-0.0259
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 012/50 | TrainLoss=0.178533 | ValLoss=0.347761 | ValRMSE=1.213156 | MAE=0.300482 | R2=-0.0254
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 013/50 | TrainLoss=0.176194 | ValLoss=0.338247 | ValRMSE=1.210522 | MAE=0.296692 | R2=-0.0209
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 014/50 | TrainLoss=0.174031 | ValLoss=0.334771 | ValRMSE=1.209965 | MAE=0.294709 | R2=-0.0200
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 015/50 | TrainLoss=0.172688 | ValLoss=0.335644 | ValRMSE=1.210212 | MAE=0.294752 | R2=-0.0204
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 016/50 | TrainLoss=0.171162 | ValLoss=0.329689 | ValRMSE=1.209209 | MAE=0.291751 | R2=-0.0187
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 017/50 | TrainLoss=0.170846 | ValLoss=0.324724 | ValRMSE=1.208218 | MAE=0.289476 | R2=-0.0171
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 018/50 | TrainLoss=0.169686 | ValLoss=0.323465 | ValRMSE=1.208290 | MAE=0.288775 | R2=-0.0172
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 019/50 | TrainLoss=0.167433 | ValLoss=0.319682 | ValRMSE=1.206592 | MAE=0.286899 | R2=-0.0143
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 020/50 | TrainLoss=0.166694 | ValLoss=0.318757 | ValRMSE=1.207166 | MAE=0.285868 | R2=-0.0153
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 021/50 | TrainLoss=0.166358 | ValLoss=0.314756 | ValRMSE=1.205905 | MAE=0.284166 | R2=-0.0132
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 022/50 | TrainLoss=0.166962 | ValLoss=0.313262 | ValRMSE=1.205404 | MAE=0.283004 | R2=-0.0123
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 023/50 | TrainLoss=0.165067 | ValLoss=0.315264 | ValRMSE=1.206487 | MAE=0.283750 | R2=-0.0142
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 024/50 | TrainLoss=0.165389 | ValLoss=0.313142 | ValRMSE=1.206172 | MAE=0.282287 | R2=-0.0136
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 025/50 | TrainLoss=0.164006 | ValLoss=0.309704 | ValRMSE=1.204581 | MAE=0.281014 | R2=-0.0109
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 026/50 | TrainLoss=0.163825 | ValLoss=0.305437 | ValRMSE=1.203538 | MAE=0.279337 | R2=-0.0092
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 027/50 | TrainLoss=0.163986 | ValLoss=0.309594 | ValRMSE=1.204743 | MAE=0.280263 | R2=-0.0112
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 028/50 | TrainLoss=0.162709 | ValLoss=0.306303 | ValRMSE=1.204088 | MAE=0.278969 | R2=-0.0101
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 029/50 | TrainLoss=0.162336 | ValLoss=0.307706 | ValRMSE=1.204596 | MAE=0.279372 | R2=-0.0110
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 030/50 | TrainLoss=0.161772 | ValLoss=0.305083 | ValRMSE=1.203683 | MAE=0.277733 | R2=-0.0094
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 031/50 | TrainLoss=0.161416 | ValLoss=0.303880 | ValRMSE=1.203226 | MAE=0.277342 | R2=-0.0087
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 032/50 | TrainLoss=0.161059 | ValLoss=0.306844 | ValRMSE=1.204629 | MAE=0.278236 | R2=-0.0110
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 033/50 | TrainLoss=0.162208 | ValLoss=0.303542 | ValRMSE=1.203293 | MAE=0.276701 | R2=-0.0088
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 034/50 | TrainLoss=0.161113 | ValLoss=0.305154 | ValRMSE=1.203649 | MAE=0.277281 | R2=-0.0094
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 035/50 | TrainLoss=0.160648 | ValLoss=0.302155 | ValRMSE=1.202988 | MAE=0.275680 | R2=-0.0083
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 036/50 | TrainLoss=0.160560 | ValLoss=0.304936 | ValRMSE=1.204054 | MAE=0.276824 | R2=-0.0101
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 037/50 | TrainLoss=0.160621 | ValLoss=0.299099 | ValRMSE=1.201238 | MAE=0.275043 | R2=-0.0053
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 038/50 | TrainLoss=0.159896 | ValLoss=0.304533 | ValRMSE=1.204251 | MAE=0.276006 | R2=-0.0104
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 039/50 | TrainLoss=0.160032 | ValLoss=0.304549 | ValRMSE=1.203285 | MAE=0.277057 | R2=-0.0088
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 040/50 | TrainLoss=0.158658 | ValLoss=0.304298 | ValRMSE=1.203118 | MAE=0.276890 | R2=-0.0085
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 041/50 | TrainLoss=0.159762 | ValLoss=0.303149 | ValRMSE=1.203607 | MAE=0.275675 | R2=-0.0093
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 042/50 | TrainLoss=0.159146 | ValLoss=0.298692 | ValRMSE=1.201031 | MAE=0.274133 | R2=-0.0050
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 043/50 | TrainLoss=0.159408 | ValLoss=0.297040 | ValRMSE=1.200969 | MAE=0.273464 | R2=-0.0049
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 044/50 | TrainLoss=0.159643 | ValLoss=0.299010 | ValRMSE=1.201303 | MAE=0.274023 | R2=-0.0055
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 045/50 | TrainLoss=0.158637 | ValLoss=0.301942 | ValRMSE=1.202677 | MAE=0.275297 | R2=-0.0078
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 046/50 | TrainLoss=0.158071 | ValLoss=0.300161 | ValRMSE=1.202315 | MAE=0.273877 | R2=-0.0071
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 047/50 | TrainLoss=0.158723 | ValLoss=0.301445 | ValRMSE=1.202799 | MAE=0.274687 | R2=-0.0080
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 048/50 | TrainLoss=0.158445 | ValLoss=0.301304 | ValRMSE=1.201812 | MAE=0.274788 | R2=-0.0063
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 049/50 | TrainLoss=0.157903 | ValLoss=0.299610 | ValRMSE=1.201961 | MAE=0.273634 | R2=-0.0066
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 050/50 | TrainLoss=0.158682 | ValLoss=0.301705 | ValRMSE=1.202517 | MAE=0.274368 | R2=-0.0075

========== STAGE 2: FINE-TUNE FULL MODEL ==========
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 001/500 | TrainLoss=0.156288 | ValLoss=0.290046 | ValRMSE=1.197323 | MAE=0.269479 | R2=0.0012
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 002/500 | TrainLoss=0.151059 | ValLoss=0.288916 | ValRMSE=1.195666 | MAE=0.268742 | R2=0.0040
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 003/500 | TrainLoss=0.147972 | ValLoss=0.287892 | ValRMSE=1.193720 | MAE=0.268038 | R2=0.0072
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 004/500 | TrainLoss=0.144008 | ValLoss=0.284367 | ValRMSE=1.191076 | MAE=0.266053 | R2=0.0116
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 005/500 | TrainLoss=0.141306 | ValLoss=0.282607 | ValRMSE=1.188852 | MAE=0.265240 | R2=0.0153
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 006/500 | TrainLoss=0.138900 | ValLoss=0.281654 | ValRMSE=1.186604 | MAE=0.264981 | R2=0.0190
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 007/500 | TrainLoss=0.136417 | ValLoss=0.276844 | ValRMSE=1.183657 | MAE=0.262995 | R2=0.0239
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 008/500 | TrainLoss=0.134281 | ValLoss=0.278279 | ValRMSE=1.183025 | MAE=0.263513 | R2=0.0249
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 009/500 | TrainLoss=0.132489 | ValLoss=0.276037 | ValRMSE=1.180754 | MAE=0.262615 | R2=0.0286
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 010/500 | TrainLoss=0.129449 | ValLoss=0.276267 | ValRMSE=1.179213 | MAE=0.262483 | R2=0.0312
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 011/500 | TrainLoss=0.128225 | ValLoss=0.277011 | ValRMSE=1.177908 | MAE=0.262849 | R2=0.0333
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 012/500 | TrainLoss=0.125444 | ValLoss=0.273128 | ValRMSE=1.174645 | MAE=0.261343 | R2=0.0387
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 013/500 | TrainLoss=0.124414 | ValLoss=0.268399 | ValRMSE=1.170038 | MAE=0.259689 | R2=0.0462
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 014/500 | TrainLoss=0.121464 | ValLoss=0.268996 | ValRMSE=1.168429 | MAE=0.259465 | R2=0.0488
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 015/500 | TrainLoss=0.119196 | ValLoss=0.263826 | ValRMSE=1.164886 | MAE=0.256490 | R2=0.0546
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 016/500 | TrainLoss=0.116976 | ValLoss=0.268483 | ValRMSE=1.165229 | MAE=0.258453 | R2=0.0540
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 017/500 | TrainLoss=0.114624 | ValLoss=0.266786 | ValRMSE=1.161176 | MAE=0.258235 | R2=0.0606
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 018/500 | TrainLoss=0.112679 | ValLoss=0.260933 | ValRMSE=1.155845 | MAE=0.255743 | R2=0.0692
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 019/500 | TrainLoss=0.111590 | ValLoss=0.260752 | ValRMSE=1.153151 | MAE=0.255606 | R2=0.0735
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 020/500 | TrainLoss=0.108265 | ValLoss=0.260219 | ValRMSE=1.149682 | MAE=0.255416 | R2=0.0791
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 021/500 | TrainLoss=0.106722 | ValLoss=0.256538 | ValRMSE=1.143987 | MAE=0.254124 | R2=0.0882
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 022/500 | TrainLoss=0.102780 | ValLoss=0.252197 | ValRMSE=1.139808 | MAE=0.252815 | R2=0.0948
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 023/500 | TrainLoss=0.100854 | ValLoss=0.254587 | ValRMSE=1.138063 | MAE=0.253621 | R2=0.0976
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 024/500 | TrainLoss=0.097037 | ValLoss=0.250682 | ValRMSE=1.132591 | MAE=0.251325 | R2=0.1063
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 025/500 | TrainLoss=0.096281 | ValLoss=0.250773 | ValRMSE=1.129497 | MAE=0.251530 | R2=0.1112
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 026/500 | TrainLoss=0.092912 | ValLoss=0.246935 | ValRMSE=1.125121 | MAE=0.249065 | R2=0.1180
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 027/500 | TrainLoss=0.091582 | ValLoss=0.244355 | ValRMSE=1.115802 | MAE=0.247369 | R2=0.1326
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 028/500 | TrainLoss=0.085873 | ValLoss=0.240029 | ValRMSE=1.110177 | MAE=0.245108 | R2=0.1413
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 029/500 | TrainLoss=0.085010 | ValLoss=0.242581 | ValRMSE=1.111551 | MAE=0.245002 | R2=0.1392
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 030/500 | TrainLoss=0.081700 | ValLoss=0.235975 | ValRMSE=1.103757 | MAE=0.240991 | R2=0.1512
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 031/500 | TrainLoss=0.081974 | ValLoss=0.241234 | ValRMSE=1.099888 | MAE=0.243393 | R2=0.1571
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 032/500 | TrainLoss=0.077538 | ValLoss=0.231803 | ValRMSE=1.088634 | MAE=0.237774 | R2=0.1743
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 033/500 | TrainLoss=0.075326 | ValLoss=0.234276 | ValRMSE=1.087761 | MAE=0.237432 | R2=0.1756
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 034/500 | TrainLoss=0.072945 | ValLoss=0.234565 | ValRMSE=1.083080 | MAE=0.236201 | R2=0.1827
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 035/500 | TrainLoss=0.070697 | ValLoss=0.236858 | ValRMSE=1.083337 | MAE=0.236511 | R2=0.1823
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 036/500 | TrainLoss=0.070127 | ValLoss=0.233319 | ValRMSE=1.081112 | MAE=0.236403 | R2=0.1857
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 037/500 | TrainLoss=0.070845 | ValLoss=0.230059 | ValRMSE=1.071214 | MAE=0.231521 | R2=0.2005
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 038/500 | TrainLoss=0.065012 | ValLoss=0.224953 | ValRMSE=1.065340 | MAE=0.228195 | R2=0.2093
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 039/500 | TrainLoss=0.064150 | ValLoss=0.241833 | ValRMSE=1.077751 | MAE=0.234283 | R2=0.1907
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 040/500 | TrainLoss=0.066253 | ValLoss=0.240384 | ValRMSE=1.075872 | MAE=0.234704 | R2=0.1935
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 041/500 | TrainLoss=0.063611 | ValLoss=0.236596 | ValRMSE=1.066365 | MAE=0.234673 | R2=0.2077
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 042/500 | TrainLoss=0.056438 | ValLoss=0.231081 | ValRMSE=1.062637 | MAE=0.231380 | R2=0.2133
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 043/500 | TrainLoss=0.058584 | ValLoss=0.233575 | ValRMSE=1.060295 | MAE=0.232294 | R2=0.2167
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 044/500 | TrainLoss=0.061579 | ValLoss=0.232211 | ValRMSE=1.061055 | MAE=0.231822 | R2=0.2156
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 045/500 | TrainLoss=0.059296 | ValLoss=0.229742 | ValRMSE=1.060123 | MAE=0.230106 | R2=0.2170
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 046/500 | TrainLoss=0.059185 | ValLoss=0.234189 | ValRMSE=1.059612 | MAE=0.233994 | R2=0.2177
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 047/500 | TrainLoss=0.061714 | ValLoss=0.228810 | ValRMSE=1.061722 | MAE=0.230090 | R2=0.2146
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 048/500 | TrainLoss=0.051274 | ValLoss=0.224852 | ValRMSE=1.056821 | MAE=0.228497 | R2=0.2219
  >>> 保存最佳回归模型
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 049/500 | TrainLoss=0.052930 | ValLoss=0.228396 | ValRMSE=1.064051 | MAE=0.235283 | R2=0.2112
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 050/500 | TrainLoss=0.052115 | ValLoss=0.225884 | ValRMSE=1.063161 | MAE=0.230020 | R2=0.2125
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 051/500 | TrainLoss=0.049166 | ValLoss=0.224303 | ValRMSE=1.060523 | MAE=0.230632 | R2=0.2164
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 052/500 | TrainLoss=0.045834 | ValLoss=0.227576 | ValRMSE=1.065095 | MAE=0.234631 | R2=0.2096
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 053/500 | TrainLoss=0.051514 | ValLoss=0.231042 | ValRMSE=1.070151 | MAE=0.235459 | R2=0.2021
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 054/500 | TrainLoss=0.051686 | ValLoss=0.227620 | ValRMSE=1.069924 | MAE=0.234676 | R2=0.2024
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 055/500 | TrainLoss=0.047073 | ValLoss=0.227484 | ValRMSE=1.070927 | MAE=0.235674 | R2=0.2009
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 056/500 | TrainLoss=0.050752 | ValLoss=0.227170 | ValRMSE=1.062406 | MAE=0.233459 | R2=0.2136
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 057/500 | TrainLoss=0.045804 | ValLoss=0.227346 | ValRMSE=1.071531 | MAE=0.236586 | R2=0.2000
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 058/500 | TrainLoss=0.045715 | ValLoss=0.236739 | ValRMSE=1.072705 | MAE=0.239741 | R2=0.1983
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 059/500 | TrainLoss=0.044548 | ValLoss=0.230375 | ValRMSE=1.077677 | MAE=0.239866 | R2=0.1908
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 060/500 | TrainLoss=0.044177 | ValLoss=0.227658 | ValRMSE=1.075182 | MAE=0.238387 | R2=0.1946
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 061/500 | TrainLoss=0.045399 | ValLoss=0.231086 | ValRMSE=1.081618 | MAE=0.239856 | R2=0.1849
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 062/500 | TrainLoss=0.039906 | ValLoss=0.232216 | ValRMSE=1.080096 | MAE=0.241733 | R2=0.1872
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 063/500 | TrainLoss=0.043724 | ValLoss=0.231907 | ValRMSE=1.078026 | MAE=0.239012 | R2=0.1903
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 064/500 | TrainLoss=0.043516 | ValLoss=0.231736 | ValRMSE=1.077530 | MAE=0.238117 | R2=0.1911
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 065/500 | TrainLoss=0.043752 | ValLoss=0.232734 | ValRMSE=1.086115 | MAE=0.241689 | R2=0.1781
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 066/500 | TrainLoss=0.040142 | ValLoss=0.234955 | ValRMSE=1.092285 | MAE=0.244131 | R2=0.1688
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 067/500 | TrainLoss=0.041862 | ValLoss=0.231768 | ValRMSE=1.096958 | MAE=0.242489 | R2=0.1616
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 068/500 | TrainLoss=0.040182 | ValLoss=0.230655 | ValRMSE=1.092360 | MAE=0.240895 | R2=0.1686
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 069/500 | TrainLoss=0.038414 | ValLoss=0.231996 | ValRMSE=1.103303 | MAE=0.245395 | R2=0.1519
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 070/500 | TrainLoss=0.042335 | ValLoss=0.236214 | ValRMSE=1.097970 | MAE=0.245644 | R2=0.1601
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 071/500 | TrainLoss=0.045617 | ValLoss=0.232800 | ValRMSE=1.109896 | MAE=0.247369 | R2=0.1417
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 072/500 | TrainLoss=0.040744 | ValLoss=0.243178 | ValRMSE=1.129818 | MAE=0.257221 | R2=0.1106
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 073/500 | TrainLoss=0.041506 | ValLoss=0.235862 | ValRMSE=1.118627 | MAE=0.252344 | R2=0.1282
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 074/500 | TrainLoss=0.033800 | ValLoss=0.231426 | ValRMSE=1.130173 | MAE=0.251423 | R2=0.1101
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 075/500 | TrainLoss=0.043097 | ValLoss=0.237928 | ValRMSE=1.099873 | MAE=0.246942 | R2=0.1572
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 076/500 | TrainLoss=0.036556 | ValLoss=0.237107 | ValRMSE=1.137731 | MAE=0.256769 | R2=0.0981
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 077/500 | TrainLoss=0.035233 | ValLoss=0.238008 | ValRMSE=1.115762 | MAE=0.252671 | R2=0.1326
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 078/500 | TrainLoss=0.037730 | ValLoss=0.235589 | ValRMSE=1.103287 | MAE=0.247093 | R2=0.1519
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 079/500 | TrainLoss=0.039119 | ValLoss=0.237172 | ValRMSE=1.103712 | MAE=0.248720 | R2=0.1513
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 080/500 | TrainLoss=0.037832 | ValLoss=0.238298 | ValRMSE=1.124120 | MAE=0.253505 | R2=0.1196
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 081/500 | TrainLoss=0.039968 | ValLoss=0.239962 | ValRMSE=1.115936 | MAE=0.252678 | R2=0.1324
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 082/500 | TrainLoss=0.035452 | ValLoss=0.232534 | ValRMSE=1.124129 | MAE=0.251507 | R2=0.1196
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 083/500 | TrainLoss=0.034572 | ValLoss=0.229839 | ValRMSE=1.118901 | MAE=0.249789 | R2=0.1278
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 084/500 | TrainLoss=0.035396 | ValLoss=0.242315 | ValRMSE=1.121473 | MAE=0.255195 | R2=0.1237
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 085/500 | TrainLoss=0.033285 | ValLoss=0.237162 | ValRMSE=1.131878 | MAE=0.253250 | R2=0.1074
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 086/500 | TrainLoss=0.032324 | ValLoss=0.234529 | ValRMSE=1.127863 | MAE=0.250295 | R2=0.1137
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 087/500 | TrainLoss=0.031996 | ValLoss=0.237260 | ValRMSE=1.108539 | MAE=0.247528 | R2=0.1438
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 088/500 | TrainLoss=0.033685 | ValLoss=0.233439 | ValRMSE=1.123331 | MAE=0.250508 | R2=0.1208
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 089/500 | TrainLoss=0.032768 | ValLoss=0.237547 | ValRMSE=1.145281 | MAE=0.257524 | R2=0.0861
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 090/500 | TrainLoss=0.030831 | ValLoss=0.234895 | ValRMSE=1.119264 | MAE=0.252044 | R2=0.1272
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 091/500 | TrainLoss=0.034580 | ValLoss=0.241205 | ValRMSE=1.122338 | MAE=0.255723 | R2=0.1224
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 092/500 | TrainLoss=0.036383 | ValLoss=0.237638 | ValRMSE=1.127240 | MAE=0.253592 | R2=0.1147
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 093/500 | TrainLoss=0.034074 | ValLoss=0.237878 | ValRMSE=1.137650 | MAE=0.253394 | R2=0.0983
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 094/500 | TrainLoss=0.030980 | ValLoss=0.234282 | ValRMSE=1.111035 | MAE=0.246627 | R2=0.1400
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 095/500 | TrainLoss=0.033291 | ValLoss=0.235839 | ValRMSE=1.118221 | MAE=0.248355 | R2=0.1288
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 096/500 | TrainLoss=0.033842 | ValLoss=0.236212 | ValRMSE=1.130334 | MAE=0.249264 | R2=0.1098
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 097/500 | TrainLoss=0.032731 | ValLoss=0.231975 | ValRMSE=1.138327 | MAE=0.248310 | R2=0.0972
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 098/500 | TrainLoss=0.037812 | ValLoss=0.234766 | ValRMSE=1.108688 | MAE=0.245976 | R2=0.1436
Early Stopping

========== TEST: PURE REGRESSION ==========
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
RMSE: 1.575092
MAE: 0.267897
R2: 0.651848
Pearson: 0.898017
Spearman: 0.512477
/data/caoying/miniconda3/envs/enzymes/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Best model: /data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_reg_enz2/best_regression.pth
Scaler: /data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_reg_enz2/scaler.json
Output: /data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_reg_enz2
