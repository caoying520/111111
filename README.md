nohup: ignoring input
Using device: cuda
ESM=165, MolFormer胺=23, MolFormer酮=12
Train=4538, Val=951, Test=1028
Train Activity分布:
count    4538.000000
mean        0.806379
std         2.242486
min         0.000000
25%         0.000000
50%         0.064450
75%         0.508400
max        27.666500
Name: Activity, dtype: float64
Val Activity分布:
count    951.000000
mean       0.832335
std        2.319837
min        0.000000
25%        0.000000
50%        0.058000
75%        0.559700
max       25.850100
Name: Activity, dtype: float64
Test Activity分布:
count    1028.000000
mean        0.794268
std         1.929052
min         0.000000
25%         0.000000
50%         0.053600
75%         0.495100
max        16.698300
Name: Activity, dtype: float64
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 1/500 | Train Loss=0.2950 | Val Loss=0.1975 | MAE=0.8108 | RMSE=2.4002 | R2=-0.0716 | Spearman=0.4694 | Active Spearman=0.3618 | NDCG@100=0.4168 | Top100 Overlap=0.4000
保存最佳模型！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 2/500 | Train Loss=0.2543 | Val Loss=0.2006 | MAE=0.8095 | RMSE=2.3768 | R2=-0.0508 | Spearman=0.4736 | Active Spearman=0.4032 | NDCG@100=0.4191 | Top100 Overlap=0.3800
保存最佳模型！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 3/500 | Train Loss=0.2269 | Val Loss=0.2329 | MAE=0.8594 | RMSE=2.3413 | R2=-0.0197 | Spearman=0.5258 | Active Spearman=0.4177 | NDCG@100=0.5235 | Top100 Overlap=0.4900
保存最佳模型！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 4/500 | Train Loss=0.2026 | Val Loss=0.2058 | MAE=0.8134 | RMSE=2.2860 | R2=0.0279 | Spearman=0.5797 | Active Spearman=0.4542 | NDCG@100=0.5424 | Top100 Overlap=0.5200
保存最佳模型！
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 5/500 | Train Loss=0.1854 | Val Loss=0.1840 | MAE=0.7830 | RMSE=2.2430 | R2=0.0641 | Spearman=0.6069 | Active Spearman=0.5168 | NDCG@100=0.4837 | Top100 Overlap=0.4700
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 6/500 | Train Loss=0.1710 | Val Loss=0.1599 | MAE=0.7377 | RMSE=2.2088 | R2=0.0924 | Spearman=0.6095 | Active Spearman=0.5162 | NDCG@100=0.5055 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 7/500 | Train Loss=0.1636 | Val Loss=0.1647 | MAE=0.7509 | RMSE=2.1856 | R2=0.1115 | Spearman=0.6561 | Active Spearman=0.5441 | NDCG@100=0.4749 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 8/500 | Train Loss=0.1609 | Val Loss=0.1644 | MAE=0.7424 | RMSE=2.2047 | R2=0.0959 | Spearman=0.6040 | Active Spearman=0.5335 | NDCG@100=0.4879 | Top100 Overlap=0.4900
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 9/500 | Train Loss=0.1463 | Val Loss=0.2329 | MAE=0.8592 | RMSE=2.3654 | R2=-0.0407 | Spearman=0.4982 | Active Spearman=0.4436 | NDCG@100=0.4861 | Top100 Overlap=0.4800
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 10/500 | Train Loss=0.1483 | Val Loss=0.1585 | MAE=0.7371 | RMSE=2.2198 | R2=0.0834 | Spearman=0.6840 | Active Spearman=0.6249 | NDCG@100=0.4883 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 11/500 | Train Loss=0.1391 | Val Loss=0.1619 | MAE=0.7455 | RMSE=2.2444 | R2=0.0630 | Spearman=0.6270 | Active Spearman=0.5451 | NDCG@100=0.4566 | Top100 Overlap=0.4600
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 12/500 | Train Loss=0.1400 | Val Loss=0.1634 | MAE=0.7454 | RMSE=2.2300 | R2=0.0749 | Spearman=0.6330 | Active Spearman=0.5611 | NDCG@100=0.4819 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 13/500 | Train Loss=0.1373 | Val Loss=0.1429 | MAE=0.7288 | RMSE=2.1200 | R2=0.1640 | Spearman=0.6855 | Active Spearman=0.5882 | NDCG@100=0.4836 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 14/500 | Train Loss=0.1275 | Val Loss=0.1461 | MAE=0.7208 | RMSE=2.1688 | R2=0.1250 | Spearman=0.6774 | Active Spearman=0.5713 | NDCG@100=0.4865 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 15/500 | Train Loss=0.1269 | Val Loss=0.1625 | MAE=0.7377 | RMSE=2.2355 | R2=0.0704 | Spearman=0.6402 | Active Spearman=0.5450 | NDCG@100=0.4980 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 16/500 | Train Loss=0.1254 | Val Loss=0.1550 | MAE=0.7368 | RMSE=2.1858 | R2=0.1113 | Spearman=0.6484 | Active Spearman=0.5496 | NDCG@100=0.4604 | Top100 Overlap=0.4800
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 17/500 | Train Loss=0.1259 | Val Loss=0.1662 | MAE=0.7464 | RMSE=2.2610 | R2=0.0491 | Spearman=0.6215 | Active Spearman=0.5121 | NDCG@100=0.4511 | Top100 Overlap=0.4700
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 18/500 | Train Loss=0.1228 | Val Loss=0.1557 | MAE=0.7252 | RMSE=2.2205 | R2=0.0829 | Spearman=0.6464 | Active Spearman=0.5397 | NDCG@100=0.4944 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 19/500 | Train Loss=0.1237 | Val Loss=0.1412 | MAE=0.7057 | RMSE=2.1439 | R2=0.1450 | Spearman=0.6885 | Active Spearman=0.5886 | NDCG@100=0.5039 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 20/500 | Train Loss=0.1178 | Val Loss=0.1465 | MAE=0.7097 | RMSE=2.1862 | R2=0.1110 | Spearman=0.6823 | Active Spearman=0.5916 | NDCG@100=0.5050 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 21/500 | Train Loss=0.1198 | Val Loss=0.1545 | MAE=0.7233 | RMSE=2.2089 | R2=0.0924 | Spearman=0.6584 | Active Spearman=0.5680 | NDCG@100=0.4952 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 22/500 | Train Loss=0.1175 | Val Loss=0.1538 | MAE=0.7242 | RMSE=2.1847 | R2=0.1122 | Spearman=0.6583 | Active Spearman=0.5518 | NDCG@100=0.4908 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 23/500 | Train Loss=0.1166 | Val Loss=0.1491 | MAE=0.7148 | RMSE=2.1691 | R2=0.1248 | Spearman=0.6764 | Active Spearman=0.5711 | NDCG@100=0.5068 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 24/500 | Train Loss=0.1159 | Val Loss=0.1494 | MAE=0.7187 | RMSE=2.1755 | R2=0.1196 | Spearman=0.6688 | Active Spearman=0.5566 | NDCG@100=0.4966 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 25/500 | Train Loss=0.1162 | Val Loss=0.1522 | MAE=0.7195 | RMSE=2.1926 | R2=0.1057 | Spearman=0.6596 | Active Spearman=0.5519 | NDCG@100=0.4993 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 26/500 | Train Loss=0.1125 | Val Loss=0.1510 | MAE=0.7196 | RMSE=2.1954 | R2=0.1035 | Spearman=0.6566 | Active Spearman=0.5479 | NDCG@100=0.4875 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 27/500 | Train Loss=0.1140 | Val Loss=0.1574 | MAE=0.7305 | RMSE=2.2106 | R2=0.0910 | Spearman=0.6503 | Active Spearman=0.5478 | NDCG@100=0.4853 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 28/500 | Train Loss=0.1145 | Val Loss=0.1498 | MAE=0.7184 | RMSE=2.1880 | R2=0.1095 | Spearman=0.6785 | Active Spearman=0.5659 | NDCG@100=0.4800 | Top100 Overlap=0.4900
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 29/500 | Train Loss=0.1150 | Val Loss=0.1546 | MAE=0.7276 | RMSE=2.2089 | R2=0.0924 | Spearman=0.6813 | Active Spearman=0.5624 | NDCG@100=0.4860 | Top100 Overlap=0.4900
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 30/500 | Train Loss=0.1113 | Val Loss=0.1577 | MAE=0.7316 | RMSE=2.2168 | R2=0.0859 | Spearman=0.6612 | Active Spearman=0.5506 | NDCG@100=0.5011 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 31/500 | Train Loss=0.1111 | Val Loss=0.1508 | MAE=0.7175 | RMSE=2.1901 | R2=0.1078 | Spearman=0.6654 | Active Spearman=0.5711 | NDCG@100=0.4894 | Top100 Overlap=0.5100
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 32/500 | Train Loss=0.1068 | Val Loss=0.1505 | MAE=0.7167 | RMSE=2.1864 | R2=0.1108 | Spearman=0.6635 | Active Spearman=0.5701 | NDCG@100=0.4917 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 33/500 | Train Loss=0.1115 | Val Loss=0.1468 | MAE=0.7110 | RMSE=2.1749 | R2=0.1201 | Spearman=0.6713 | Active Spearman=0.5759 | NDCG@100=0.5022 | Top100 Overlap=0.5300
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 34/500 | Train Loss=0.1079 | Val Loss=0.1478 | MAE=0.7135 | RMSE=2.1840 | R2=0.1127 | Spearman=0.6666 | Active Spearman=0.5687 | NDCG@100=0.4928 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 35/500 | Train Loss=0.1095 | Val Loss=0.1524 | MAE=0.7216 | RMSE=2.2003 | R2=0.0995 | Spearman=0.6606 | Active Spearman=0.5666 | NDCG@100=0.4842 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 36/500 | Train Loss=0.1048 | Val Loss=0.1522 | MAE=0.7207 | RMSE=2.1955 | R2=0.1034 | Spearman=0.6583 | Active Spearman=0.5686 | NDCG@100=0.5037 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 37/500 | Train Loss=0.1051 | Val Loss=0.1504 | MAE=0.7165 | RMSE=2.1834 | R2=0.1132 | Spearman=0.6644 | Active Spearman=0.5721 | NDCG@100=0.5031 | Top100 Overlap=0.5300
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 38/500 | Train Loss=0.1089 | Val Loss=0.1501 | MAE=0.7167 | RMSE=2.1819 | R2=0.1144 | Spearman=0.6664 | Active Spearman=0.5745 | NDCG@100=0.5015 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 39/500 | Train Loss=0.1107 | Val Loss=0.1489 | MAE=0.7135 | RMSE=2.1818 | R2=0.1146 | Spearman=0.6681 | Active Spearman=0.5817 | NDCG@100=0.5017 | Top100 Overlap=0.5300
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 40/500 | Train Loss=0.1044 | Val Loss=0.1476 | MAE=0.7121 | RMSE=2.1761 | R2=0.1191 | Spearman=0.6765 | Active Spearman=0.5783 | NDCG@100=0.4963 | Top100 Overlap=0.5200
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 41/500 | Train Loss=0.1080 | Val Loss=0.1531 | MAE=0.7212 | RMSE=2.1990 | R2=0.1005 | Spearman=0.6667 | Active Spearman=0.5705 | NDCG@100=0.4925 | Top100 Overlap=0.5000
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 42/500 | Train Loss=0.1069 | Val Loss=0.1517 | MAE=0.7196 | RMSE=2.1943 | R2=0.1043 | Spearman=0.6720 | Active Spearman=0.5640 | NDCG@100=0.4867 | Top100 Overlap=0.4900
/data/AIPD/miniconda3/envs/R2/lib/python3.11/site-packages/torch/nn/modules/transformer.py:562: UserWarning: Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly. (Triggered internally at ../aten/src/ATen/native/transformers/attention.cpp:150.)
  return torch._transformer_encoder_layer_fwd(
Epoch 43/500 | Train Loss=0.1063 | Val Loss=0.1501 | MAE=0.7161 | RMSE=2.1895 | R2=0.1083 | Spearman=0.6739 | Active Spearman=0.5665 | NDCG@100=0.4909 | Top100 Overlap=0.5100
