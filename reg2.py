# -*- coding: utf-8 -*-
"""
Activity 纯回归模型预测脚本
模型结构、浓度预处理和输出反变换与训练脚本
activity_regression_overlap_weight_concentration.py 保持一致。

输入 CSV 至少需要：
    Name, SM, SM_norm, Enz_norm
其中 SM_norm / Enz_norm 的列名和浓度处理方式以训练生成的 scaler.json 为准。
如果训练时 CONC_MODE="raw_gl"，输入 CSV 应提供训练时对应的原始 g/L 浓度列。
不需要 Activity 标签列。
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==================== 1. 路径与参数：按实际情况修改 ====================
MODEL_DIR = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/KRD/13K/model_Activity_reg_enz/split2/split_去重复酶权重_加浓度"
)

BEST_MODEL = os.path.join(MODEL_DIR, "best_regression.pth")
SCALER_JSON = os.path.join(MODEL_DIR, "scaler.json")

# 待预测数据：至少包含 Name、SM 和训练所需的浓度列
INPUT_CSV = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/predict_input.csv"

ESM_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/KRD/13K/esm_144KRD.pkl"
)
MOL_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/KRD/13K/SM96_std.pkl"
)

OUTPUT_CSV = os.path.join(MODEL_DIR, "predicted_activity.csv")

BATCH_SIZE = 128
NUM_WORKERS = 4


# ==================== 2. 模型：必须与训练代码完全一致 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 256)
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 256)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.esm_encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )

        self.sm_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128)
        )
        self.enz_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128)
        )

        self.fc1 = nn.Linear(768, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, esm, mask, mol, sm, enz):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)

        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1e-8)

        enz_feat = self.enz_proj(enz.unsqueeze(-1))
        mol_feat = self.mol_proj(mol)
        sm_feat = self.sm_proj(sm.unsqueeze(-1))

        x = torch.cat([pooled, enz_feat, mol_feat, sm_feat], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))

        residual = x
        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


# ==================== 3. 特征读取与数据集 ====================
def load_pickle(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到特征文件：{path}")
    with open(path, "rb") as f:
        return pickle.load(f)


class PredictionDataset(Dataset):
    def __init__(self, df, esm_dict, mol_dict):
        self.df = df.reset_index(drop=True)
        self.esm_dict = esm_dict
        self.mol_dict = mol_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        name, sm_id = str(row["Name"]), str(row["SM"])

        if name not in self.esm_dict:
            raise KeyError(f"ESM 特征中不存在酶 Name={name}")
        if sm_id not in self.mol_dict:
            raise KeyError(f"MolFormer 特征中不存在底物 SM={sm_id}")

        feature = self.esm_dict[name]
        emb = torch.as_tensor(np.asarray(feature["emb"]), dtype=torch.float32)
        mask = torch.as_tensor(
            np.asarray(feature["mask"]), dtype=torch.bool
        ).reshape(-1)

        n = min(len(emb), len(mask))
        emb, mask = emb[:n], mask[:n]
        if n == 0:
            raise ValueError(f"酶 {name} 的 ESM 特征长度为 0")

        mol = torch.as_tensor(
            np.asarray(self.mol_dict[sm_id]), dtype=torch.float32
        ).reshape(-1)

        if emb.ndim != 2 or emb.shape[1] != 1280:
            raise ValueError(
                f"酶 {name} 的 ESM 维度错误，期望 [L,1280]，实际 {tuple(emb.shape)}"
            )
        if mol.numel() != 768:
            raise ValueError(
                f"底物 {sm_id} 的 MolFormer 维度错误，期望 768，实际 {mol.numel()}"
            )

        return (
            emb, mask, mol,
            torch.tensor(float(row["_sm_input"]), dtype=torch.float32),
            torch.tensor(float(row["_enz_input"]), dtype=torch.float32),
            str(row["idx"]), name, sm_id
        )


def collate_fn(batch):
    max_len = max(x[0].shape[0] for x in batch)
    embs, masks = [], []

    for item in batch:
        emb, mask = item[0], item[1]
        n = min(len(emb), len(mask))
        emb, mask = emb[:n], mask[:n]
        pad_len = max_len - n

        if pad_len > 0:
            emb = torch.cat(
                [emb, torch.zeros(pad_len, emb.shape[1], dtype=emb.dtype)],
                dim=0
            )
            mask = torch.cat(
                [mask, torch.zeros(pad_len, dtype=torch.bool)],
                dim=0
            )

        embs.append(emb)
        masks.append(mask)

    return (
        torch.stack(embs),
        torch.stack(masks),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        [x[5] for x in batch],
        [x[6] for x in batch],
        [x[7] for x in batch]
    )


# ==================== 4. 加载模型与预处理参数 ====================
def load_model(model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到训练好的模型：{model_path}")

    model = EnzymeSubstrateRegressor().to(device)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_scaler(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到训练时的 scaler.json：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(df, scaler):
    """严格复用训练时的浓度列名、模式和训练集统计量。"""
    sm_col = scaler["SM_COL"]
    enz_col = scaler["ENZ_COL"]
    mode = scaler["concentration_mode"]

    required = ["Name", "SM", sm_col, enz_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"预测 CSV 缺少必要列：{missing}")

    df = df.copy()
    for col in ["Name", "SM"]:
        df[col] = df[col].astype(str).str.strip()

    if "idx" not in df.columns:
        df.insert(0, "idx", [f"predict_{i+1}" for i in range(len(df))])
    else:
        df["idx"] = df["idx"].astype(str).str.strip()

    df[sm_col] = pd.to_numeric(df[sm_col], errors="coerce")
    df[enz_col] = pd.to_numeric(df[enz_col], errors="coerce")

    if df[sm_col].isna().any() or df[enz_col].isna().any():
        bad = df[df[sm_col].isna() | df[enz_col].isna()].index.tolist()[:10]
        raise ValueError(f"浓度列存在空值或非数值，示例行索引：{bad}")

    if (df[sm_col] < 0).any() or (df[enz_col] < 0).any():
        raise ValueError("浓度不能为负数")

    if mode == "normalized":
        # 训练时 normalized 模式直接使用已有标准化浓度，不再二次标准化。
        df["_sm_input"] = df[sm_col].astype(np.float32)
        df["_enz_input"] = df[enz_col].astype(np.float32)

    elif mode == "raw_gl":
        # 训练时 raw_gl 模式：log1p 后使用训练集均值和标准差做 z-score。
        sm_mean, sm_std = scaler["SM_mean"], scaler["SM_std"]
        enz_mean, enz_std = scaler["Enz_mean"], scaler["Enz_std"]

        if sm_mean is None or sm_std is None or enz_mean is None or enz_std is None:
            raise ValueError("scaler.json 中缺少 raw_gl 浓度标准化参数")

        sm_log = np.log1p(df[sm_col].to_numpy(dtype=float))
        enz_log = np.log1p(df[enz_col].to_numpy(dtype=float))

        df["_sm_input"] = ((sm_log - sm_mean) / sm_std).astype(np.float32)
        df["_enz_input"] = ((enz_log - enz_mean) / enz_std).astype(np.float32)

    else:
        raise ValueError(f"未知 concentration_mode：{mode}")

    return df


# ==================== 5. 批量预测 ====================
@torch.no_grad()
def run_prediction(model, loader, device, scaler):
    pred_scaled, meta = [], []

    for batch in loader:
        esm, mask, mol, sm, enz = [
            x.to(device, non_blocking=True) for x in batch[:5]
        ]

        pred = model(esm, mask, mol, sm, enz)
        pred_scaled.extend(pred.detach().cpu().numpy().tolist())
        meta.extend(zip(*batch[5:8]))

    # 复用训练时 Activity 的 log1p + train-set z-score 反变换。
    y_mean = float(scaler["activity_mean"])
    y_std = float(scaler["activity_std"])
    pred_activity = np.expm1(
        np.asarray(pred_scaled, dtype=np.float64) * y_std + y_mean
    )

    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["Pred_Activity"] = pred_activity

    # 同一底物下不同酶的相对排名；排名 1 表示预测活性最高。
    result["Pred_Rank_within_SM"] = (
        result.groupby("SM")["Pred_Activity"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result["Pred_Rank_Global"] = (
        result["Pred_Activity"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return result


def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"请先将 INPUT_CSV 修改为待预测 CSV 的真实路径：{INPUT_CSV}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    scaler = load_scaler(SCALER_JSON)
    esm_dict = load_pickle(ESM_PKL)
    mol_dict = load_pickle(MOL_PKL)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if df.empty:
        raise ValueError("输入 CSV 没有数据")

    df = prepare_input(df, scaler)

    missing_enz = sorted(set(df["Name"]) - set(esm_dict))
    missing_sm = sorted(set(df["SM"]) - set(mol_dict))
    if missing_enz:
        raise KeyError(
            f"以下酶缺少 ESM 特征，共 {len(missing_enz)} 个；示例：{missing_enz[:10]}"
        )
    if missing_sm:
        raise KeyError(
            f"以下底物缺少 MolFormer 特征，共 {len(missing_sm)} 个；示例：{missing_sm[:10]}"
        )

    dataset = PredictionDataset(df, esm_dict, mol_dict)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    model = load_model(BEST_MODEL, device)
    pred_df = run_prediction(model, loader, device, scaler)

    # 将输入 CSV 的其他信息列一并保留，并按 idx 对齐预测结果。
    output = df.drop(columns=["_sm_input", "_enz_input"]).merge(
        pred_df, on=["idx", "Name", "SM"], how="left", validate="one_to_one"
    )
    output = output.sort_values(
        "Pred_Activity", ascending=False
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n========== PREDICTION FINISHED ==========")
    print("预测样本数：", len(output))
    print("预测活性范围：",
          float(output["Pred_Activity"].min()),
          "—",
          float(output["Pred_Activity"].max()))
    print("结果文件：", OUTPUT_CSV)
    print("\nTop 10:")
    print(output[["idx", "Name", "SM", "Pred_Activity",
                  "Pred_Rank_within_SM", "Pred_Rank_Global"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
