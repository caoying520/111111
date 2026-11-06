# train.py
# 三模态全局特征融合模型的 10-fold StratifiedGroupKFold 训练脚本
# 数据表结构沿用原脚本：
#   5ml_homology_clusters_unified_train.csv
#   x_5ml_esm2_dataset_with_IDs.csv
#   x_5ml_unirep_dataset_with_IDs.csv
#   x_5ml_protT5_dataset_with_IDs.csv
#
# 特征 CSV：第一列为 Sequence_ID，后续列为浮点特征。
# 聚类 CSV：至少包含 Sequence_ID 和 Sim_40（或 --sim_threshold 指定列）。
# Sequence_ID 最后一个下划线后的数字必须是类别标签 0/1/2。

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report
)

from Model import GlobalFeatureFusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="/data/caoying/softwares/ProtSATT/new_train/Expression/feature_train")
    p.add_argument("--save_dir", type=str,
                   default="/data/caoying/softwares/ProtSATT/new_train/Expression/result/global_fusion_3class")
    p.add_argument("--sim_threshold", type=str, default="Sim_40")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--lr_patience", type=int, default=12)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiFeatureDataset(Dataset):
    def __init__(self, esm, uni, pro, labels):
        self.esm = torch.as_tensor(esm, dtype=torch.float32)
        self.uni = torch.as_tensor(uni, dtype=torch.float32)
        self.pro = torch.as_tensor(pro, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.esm[idx], self.uni[idx], self.pro[idx], self.labels[idx]


def load_features(path, target_ids):
    # header=None：兼容原脚本无表头格式；第 1 列为 ID
    df = pd.read_csv(path, header=None, low_memory=False)
    df = df.rename(columns={0: "Sequence_ID"})
    df["Sequence_ID"] = df["Sequence_ID"].astype(str)
    if df["Sequence_ID"].duplicated().any():
        duplicated = df.loc[df["Sequence_ID"].duplicated(), "Sequence_ID"].head().tolist()
        raise ValueError(f"{os.path.basename(path)} 存在重复 Sequence_ID，例如：{duplicated}")

    df = df.set_index("Sequence_ID")
    missing = [sid for sid in target_ids if sid not in df.index]
    if missing:
        raise KeyError(f"{os.path.basename(path)} 缺少 {len(missing)} 个 ID，例如：{missing[:5]}")

    arr = df.loc[target_ids].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} 特征中存在 NaN 或 Inf")
