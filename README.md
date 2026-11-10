# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# coding: utf-8

import os, random, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, ndcg_score
)
from scipy.stats import pearsonr, spearmanr


# ========================
# 1. 固定随机种子
# ========================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ========================
# 2. 标签变换
# ========================
def transform_activity(y):
    y = np.asarray(y, dtype=np.float32)

    if np.any(~np.isfinite(y)):
        raise ValueError("Activity 中存在 NaN 或 inf，请先检查原始数据！")

    if np.any(y < 0):
        raise ValueError(
            "Activity 中存在负值，不能直接使用 log1p。请先检查实验数据。"
        )

    return np.log1p(y)


def inverse_activity(y):
    return np.expm1(np.asarray(y))


# ========================
# 3. Dataset
# ========================
class EnzymeSubstrateDataset(Dataset):

    def __init__(
        self, df, esm_embeds, mol_embeds1, mol_embeds2,
        amine2idx, ketone2idx
    ):

        self.df = df.reset_index(drop=True)

        self.idx = self.df["Name"].astype(str).tolist()
        self.smiles1 = self.df["SM_胺"].astype(str).tolist()
        self.smiles2 = self.df["SM_酮"].astype(str).tolist()

        self.labels = pd.to_numeric(
            self.df["Activity"], errors="raise"
        ).astype(float).to_numpy()

        self.labels_trans = transform_activity(self.labels)

        self.esm_embeds = esm_embeds
        self.mol_embeds1 = mol_embeds1
        self.mol_embeds2 = mol_embeds2

        self.amine2idx = amine2idx
        self.ketone2idx = ketone2idx

        self.n_amine = len(amine2idx)
        self.n_ketone = len(ketone2idx)

        # 检查特征是否存在
        for key in self.idx:
            if key not in esm_embeds:
                raise KeyError(f"ESM 特征缺失: {key}")

        for key in self.smiles1:
            if key not in mol_embeds1 or key not in amine2idx:
                raise KeyError(f"胺底物特征缺失: {key}")

        for key in self.smiles2:
            if key not in mol_embeds2 or key not in ketone2idx:
                raise KeyError(f"酮底物特征缺失: {key}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        esm_data = self.esm_embeds[self.idx[i]]

        esm_emb = torch.as_tensor(
            esm_data["emb"], dtype=torch.float32
        )

        esm_mask = torch.as_tensor(
            esm_data["mask"], dtype=torch.long
        )

        mol_feat1 = torch.as_tensor(
            self.mol_embeds1[self.smiles1[i]],
            dtype=torch.float32
        )

        mol_feat2 = torch.as_tensor(
            self.mol_embeds2[self.smiles2[i]],
            dtype=torch.float32
        )

        oh1 = F.one_hot(
            torch.tensor(self.amine2idx[self.smiles1[i]]),
            self.n_amine
        ).float()

        oh2 = F.one_hot(
            torch.tensor(self.ketone2idx[self.smiles2[i]]),
            self.n_ketone
        ).float()

        return (
            esm_emb, esm_mask, mol_feat1, mol_feat2,
            oh1, oh2,
            torch.tensor(self.labels_trans[i], dtype=torch.float32),
            torch.tensor(self.labels[i], dtype=torch.float32)
        )


# ========================
# 4. 模型
# ========================
class EnzymeSubstrateRegressor(nn.Module):

    def __init__(
        self, esm_dim=1280, mol_dim=768,
        n_amine=23, n_ketone=12,
        dropout=0.3, hidden_dim=256
    ):

        super().__init__()

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.mol_proj1 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.mol_proj2 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.esm_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )

        in_dim = 256 * 5 + n_amine + n_ketone

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm3 = nn.LayerNorm(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)

        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, esm_feat, esm_mask,
        mol_feat1, mol_feat2, oh1, oh2
    ):

        esm_feat = self.esm_proj(esm_feat)

        esm_feat = self.esm_encoder(
            esm_feat,
            src_key_padding_mask=(esm_mask == 0)
        )

        mask = esm_mask.unsqueeze(-1).float()

        esm_pooled = (
            (esm_feat * mask).sum(dim=1)
            / (mask.sum(dim=1) + 1e-8)
        )

        mol1 = self.mol_proj1(mol_feat1)
        mol2 = self.mol_proj2(mol_feat2)

        x = torch.cat([
            esm_pooled,
            mol1,
            mol2,
            esm_pooled * mol1,
            esm_pooled * mol2,
            oh1,
            oh2
        ], dim=-1)

        x = F.relu(self.norm1(self.fc1(x)))

        residual = x

        x = self.dropout(
            F.relu(self.norm2(self.fc2(x)))
        ) + residual

        x = self.dropout(
            F.relu(self.norm3(self.fc3(x)))
        )

        return self.reg_head(x).squeeze(-1)


# ========================
# 5. 高活性加权回归损失
# ========================
class HybridRegressionLoss(nn.Module):

    def __init__(
        self,
        beta=0.5,
        lambda_raw=0.25,
        raw_scale=1.0
    ):

        super().__init__()

        self.beta = beta
        self.lambda_raw = lambda_raw
        self.raw_scale = max(float(raw_scale), 1e-6)

    def forward(self, pred_log, target_log):

        target_raw = torch.expm1(target_log)

        # 高活性样本提高权重
        weight = torch.ones_like(target_raw)

        weight[target_raw >= 5] = 2.0
        weight[target_raw >= 10] = 6.0

        # Log 空间损失
        loss_log_each = F.smooth_l1_loss(
            pred_log,
            target_log,
            beta=self.beta,
            reduction="none"
        )

        loss_log = (
            loss_log_each * weight
        ).sum() / (weight.sum() + 1e-8)

        # 原始空间损失
        # 不在训练阶段 clamp pred_log，避免截断梯度
        pred_raw = torch.expm1(pred_log)

        loss_raw_each = F.smooth_l1_loss(
            pred_raw / self.raw_scale,
            target_raw / self.raw_scale,
            beta=1.0,
            reduction="none"
        )

        loss_raw = (
            loss_raw_each * weight
        ).sum() / (weight.sum() + 1e-8)

        total = loss_log + self.lambda_raw * loss_raw

        return {
            "total": total,
            "log": loss_log.detach(),
            "raw": loss_raw.detach()
        }


# ========================
# 6. 评估指标
# ========================
def safe_r2(y_true, y_pred):

    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan

    return r2_score(y_true, y_pred)


def safe_corr(func, y_true, y_pred):

    if len(y_true) < 2:
        return np.nan

    if len(np.unique(y_true)) < 2:
        return np.nan

    if len(np.unique(y_pred)) < 2:
        return np.nan

    return func(y_true, y_pred).statistic


def compute_metrics(y_true, y_pred, pred_log=None, loss=np.nan):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = safe_r2(y_true, y_pred)

    pearson = safe_corr(pearsonr, y_true, y_pred)
    spearman = safe_corr(spearmanr, y_true, y_pred)

    # 非零活性
    active = y_true > 0

    active_mae = (
        mean_absolute_error(y_true[active], y_pred[active])
        if active.sum() > 0 else np.nan
    )

    active_rmse = (
        np.sqrt(mean_squared_error(y_true[active], y_pred[active]))
        if active.sum() > 0 else np.nan
    )

    active_r2 = safe_r2(
        y_true[active], y_pred[active]
    )

    active_spearman = safe_corr(
        spearmanr, y_true[active], y_pred[active]
    )

    # 高活性
    high = y_true >= 5

    high_mae = (
        mean_absolute_error(y_true[high], y_pred[high])
        if high.sum() > 0 else np.nan
    )

    high_rmse = (
        np.sqrt(mean_squared_error(y_true[high], y_pred[high]))
        if high.sum() > 0 else np.nan
    )

    high_r2 = safe_r2(
        y_true[high], y_pred[high]
    )

    # 极高活性
    very_high = y_true >= 10

    very_high_mae = (
        mean_absolute_error(
            y_true[very_high], y_pred[very_high]
        )
        if very_high.sum() > 0 else np.nan
    )

    very_high_rmse = (
        np.sqrt(mean_squared_error(
            y_true[very_high], y_pred[very_high]
        ))
        if very_high.sum() > 0 else np.nan
    )

    # Log R2
    r2_log = np.nan

    if pred_log is not None:
        true_log = np.log1p(np.clip(y_true, 0, None))
        r2_log = safe_r2(true_log, np.asarray(pred_log))

    # Top100 只作为辅助分析，不用于训练选模
    k = min(100, len(y_true))

    true_top = np.argsort(-y_true)[:k]
    pred_top = np.argsort(-y_pred)[:k]

    overlap = (
        len(set(true_top) & set(pred_top)) / k
        if k > 0 else np.nan
    )

    return {
        "Loss": loss,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "R2_log": r2_log,
        "Pearson": pearson,
        "Spearman": spearman,
        "Active_Count": int(active.sum()),
        "Active_MAE": active_mae,
        "Active_RMSE": active_rmse,
        "Active_R2": active_r2,
        "Active_Spearman": active_spearman,
        "High_Count": int(high.sum()),
        "High_MAE": high_mae,
        "High_RMSE": high_rmse,
        "High_R2": high_r2,
        "VeryHigh_Count": int(very_high.sum()),
        "VeryHigh_MAE": very_high_mae,
        "VeryHigh_RMSE": very_high_rmse,
        "Top100_Overlap": overlap
    }


# ========================
# 7. 评估模型
# ========================
def evaluate(model, loader, device, criterion=None):

    model.eval()

    all_pred = []
    all_true = []
    all_log_pred = []

    total_loss = 0.0
    total_num = 0

    with torch.no_grad():

        for esm, mask, mol1, mol2, oh1, oh2, y, raw_y in loader:

            esm = esm.to(device)
            mask = mask.to(device)
            mol1 = mol1.to(device)
            mol2 = mol2.to(device)
            oh1 = oh1.to(device)
            oh2 = oh2.to(device)
            y = y.to(device)

            output = model(
                esm, mask, mol1, mol2, oh1, oh2
            )

            if criterion is not None:

                loss = criterion(output, y)["total"]

                total_loss += loss.item() * len(raw_y)
                total_num += len(raw_y)

            pred_log = output.cpu().numpy()

            # 只在预测阶段限制非负
            pred = inverse_activity(
                np.clip(pred_log, 0, None)
            )

            all_pred.extend(pred.tolist())
            all_true.extend(raw_y.numpy().tolist())
            all_log_pred.extend(pred_log.tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    pred_log = np.asarray(all_log_pred)

    avg_loss = (
        total_loss / total_num
        if total_num > 0 else np.nan
    )

    metrics = compute_metrics(
        y_true, y_pred, pred_log, avg_loss
    )

    predictions = {
        "true": y_true,
        "pred": y_pred,
        "pred_log": pred_log
    }

    return metrics, predictions


# ========================
# 8. 模型选择指标
# ========================
def get_selection_score(metrics):

    # 高活性样本数量不足时，避免指标过度波动
    if metrics["High_Count"] >= 5:

        score = (
            0.35 * metrics["MAE"]
            + 0.65 * metrics["High_MAE"]
        )

    else:

        score = metrics["MAE"]

    return score


# ========================
# 9. 训练函数
# ========================
def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    device,
    epochs=500,
    patience=60,
    model_dir=None
):

    os.makedirs(model_dir, exist_ok=True)

    best_score = np.inf
    patience_counter = 0

    history = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=12,
        min_lr=1e-6
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_log = 0.0
        total_raw = 0.0

        for esm, mask, mol1, mol2, oh1, oh2, y, raw_y in train_loader:

            esm = esm.to(device)
            mask = mask.to(device)
            mol1 = mol1.to(device)
            mol2 = mol2.to(device)
            oh1 = oh1.to(device)
            oh2 = oh2.to(device)
            y = y.to(device)

            output = model(
                esm, mask, mol1, mol2, oh1, oh2
            )

            parts = criterion(output, y)

            loss = parts["total"]

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_log += parts["log"].item()
            total_raw += parts["raw"].item()

        nb = max(len(train_loader), 1)

        train_loss = total_loss / nb
        train_log = total_log / nb
        train_raw = total_raw / nb

        val_metrics, _ = evaluate(
            model, val_loader, device, criterion
        )

        monitor = get_selection_score(val_metrics)

        scheduler.step(monitor)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_metrics['Loss']:.4f} | "
            f"MAE={val_metrics['MAE']:.4f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"High MAE={val_metrics['High_MAE']:.4f} | "
            f"VeryHigh MAE={val_metrics['VeryHigh_MAE']:.4f} | "
            f"Score={monitor:.4f} | LR={lr:.7f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_log": train_log,
            "train_loss_raw": train_raw,
            "selection_score": monitor,
            **{
                "val_" + k: v
                for k, v in val_metrics.items()
            }
        })

        if np.isfinite(monitor) and monitor < best_score:

            best_score = monitor
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "best_model.pth")
            )

            print("保存最佳模型！")

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch+1}"
            )

            break

    pd.DataFrame(history).to_csv(
        os.path.join(model_dir, "training_history.csv"),
        index=False
    )

    print("训练结束，最佳验证 Score：", best_score)


# ========================
# 10. 主程序
# ========================
def main():

    set_seed(123)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    base_dir = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED"
    )

    feature_dir = os.path.join(base_dir, "Feature")

    split_dir = os.path.join(
        base_dir, "split_data_Activaty_reg"
    )

    model_dir = os.path.join(
        base_dir, "model_activity_reg_highfocus"
    )

    os.makedirs(model_dir, exist_ok=True)

    # ---------- 加载特征 ----------
    with open(
        os.path.join(feature_dir, "esm_feats_emb_mask.pkl"),
        "rb"
    ) as f:

        esm_dict = pickle.load(f)

    with open(
        os.path.join(feature_dir, "SM_胺.pkl"),
        "rb"
    ) as f:

        mol_dict1 = pickle.load(f)

    with open(
        os.path.join(feature_dir, "SM_酮.pkl"),
        "rb"
    ) as f:

        mol_dict2 = pickle.load(f)

    print(
        f"ESM={len(esm_dict)}, "
        f"MolFormer胺={len(mol_dict1)}, "
        f"MolFormer酮={len(mol_dict2)}"
    )

    amine2idx = {
        s: i for i, s in enumerate(sorted(mol_dict1.keys()))
    }

    ketone2idx = {
        s: i for i, s in enumerate(sorted(mol_dict2.keys()))
    }

    # ---------- 读取 txt ----------
    def read_txt(path):

        with open(path, "r") as f:

            return [
                line.strip()
                for line in f
                if line.strip()
            ]

    train_idx = set(
        read_txt(os.path.join(split_dir, "train.txt"))
    )

    val_idx = set(
        read_txt(os.path.join(split_dir, "val.txt"))
    )

    test_idx = set(
        read_txt(os.path.join(split_dir, "test.txt"))
    )

    # 检查 txt 是否有交集
    if train_idx & val_idx:
        raise ValueError("train.txt 和 val.txt 存在重复 idx！")

    if train_idx & test_idx:
        raise ValueError("train.txt 和 test.txt 存在重复 idx！")

    if val_idx & test_idx:
        raise ValueError("val.txt 和 test.txt 存在重复 idx！")

    # ---------- 读取 CSV ----------
    df = pd.read_csv(
        os.path.join(split_dir, "train_reg.csv")
    )

    df["idx"] = df["idx"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    df["Activity"] = pd.to_numeric(
        df["Activity"], errors="raise"
    )

    if df["Activity"].isna().any():
        raise ValueError("Activity 存在缺失值！")

    if (df["Activity"] < 0).any():
        raise ValueError("Activity 存在负值，请先核查！")

    # ---------- 按 txt 划分 ----------
    df_train = df[
        df["idx"].isin(train_idx)
    ].reset_index(drop=True)

    df_val = df[
        df["idx"].isin(val_idx)
    ].reset_index(drop=True)

    df_test = df[
        df["idx"].isin(test_idx)
    ].reset_index(drop=True)

    print(
        f"Train={len(df_train)}, "
        f"Val={len(df_val)}, "
        f"Test={len(df_test)}"
    )

    # 检查 txt 是否存在无法匹配的 ID
    matched_idx = set(
        df_train["idx"]
    ) | set(
        df_val["idx"]
    ) | set(
        df_test["idx"]
    )

    missing_idx = (
        train_idx | val_idx | test_idx
    ) - matched_idx

    if missing_idx:
        raise ValueError(
            f"有 {len(missing_idx)} 个 txt ID 在 CSV 中找不到！"
        )

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError("Train、Val 或 Test 为空！")

    # Name 用作 ESM 特征键，也检查它是否确实是酶分组 ID
    train_names = set(df_train["Name"])
    val_names = set(df_val["Name"])
    test_names = set(df_test["Name"])

    if train_names & val_names:
        raise ValueError("Train 和 Val 存在相同 Name，请检查是否按酶划分！")

    if train_names & test_names:
        raise ValueError("Train 和 Test 存在相同 Name，请检查是否按酶划分！")

    if val_names & test_names:
        raise ValueError("Val 和 Test 存在相同 Name，请检查是否按酶划分！")

    # ---------- 数据分布 ----------
    def show_distribution(name, data):

        y = data["Activity"].to_numpy()

        print(f"\n{name} 分布:")
        print("总数:", len(y))
        print("零活性:", np.sum(y == 0))
        print("非零活性:", np.sum(y > 0))
        print("Activity >= 5:", np.sum(y >= 5))
        print("Activity >= 10:", np.sum(y >= 10))
        print(pd.Series(y).describe())

    show_distribution("Train", df_train)
    show_distribution("Val", df_val)
    show_distribution("Test", df_test)

    # ---------- Dataset ----------
    train_set = EnzymeSubstrateDataset(
        df_train, esm_dict, mol_dict1, mol_dict2,
        amine2idx, ketone2idx
    )

    val_set = EnzymeSubstrateDataset(
        df_val, esm_dict, mol_dict1, mol_dict2,
        amine2idx, ketone2idx
    )

    test_set = EnzymeSubstrateDataset(
        df_test, esm_dict, mol_dict1, mol_dict2,
        amine2idx, ketone2idx
    )

    batch_size = 128

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=(len(train_set) % batch_size == 1)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---------- 多 seed 集成 ----------
    seeds = [123, 42, 2024, 7, 99]

    test_true = None

    test_pred_logs = []
    test_raw_preds = []

    # 验证集只用于模型训练和选模，不用于测试集校准
    for seed in seeds:

        set_seed(seed)

        seed_model_dir = os.path.join(
            model_dir, f"seed_{seed}"
        )

        os.makedirs(seed_model_dir, exist_ok=True)

        print(
            f"\n{'='*60}\n"
            f">>> Seed {seed} 训练开始\n"
            f"{'='*60}"
        )

        model = EnzymeSubstrateRegressor(
            esm_dim=1280,
            mol_dim=768,
            n_amine=len(amine2idx),
            n_ketone=len(ketone2idx),
            dropout=0.3,
            hidden_dim=256
        ).to(device)

        raw_scale = float(
            np.std(df_train["Activity"].values)
        )

        criterion = HybridRegressionLoss(
            beta=0.5,
            lambda_raw=0.25,
            raw_scale=raw_scale
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=5e-4
        )

        train_model(
            train_loader,
            val_loader,
            model,
            optimizer,
            criterion,
            device,
            epochs=500,
            patience=60,
            model_dir=seed_model_dir
        )

        # ---------- 加载最佳模型 ----------
        best_path = os.path.join(
            seed_model_dir, "best_model.pth"
        )

        if not os.path.exists(best_path):
            raise RuntimeError(
                f"没有找到最佳模型: {best_path}"
            )

        model.load_state_dict(
            torch.load(best_path, map_location=device)
        )

        # ---------- 独立测试 ----------
        test_metrics, test_pred = evaluate(
            model, test_loader, device, criterion
        )

        print(
            f"\n[Seed {seed}] TEST: "
            f"MAE={test_metrics['MAE']:.4f} | "
            f"RMSE={test_metrics['RMSE']:.4f} | "
            f"R2={test_metrics['R2']:.4f} | "
            f"High MAE={test_metrics['High_MAE']:.4f} | "
            f"VeryHigh MAE={test_metrics['VeryHigh_MAE']:.4f}"
        )

        if test_true is None:
            test_true = test_pred["true"]

        test_pred_logs.append(
            test_pred["pred_log"]
        )

        test_raw_preds.append(
            test_pred["pred"]
        )

        del model
        torch.cuda.empty_cache()

    # ========================
    # 11. 集成预测
    # ========================
    test_pred_logs = np.asarray(test_pred_logs)

    test_raw_preds = np.asarray(test_raw_preds)

    # Log 空间平均
    ens_test_pred_log = test_pred_logs.mean(axis=0)

    # 还原原始活性
    ens_raw = inverse_activity(
        np.clip(ens_test_pred_log, 0, None)
    )

    ens_metrics = compute_metrics(
        test_true,
        ens_raw,
        ens_test_pred_log
    )

    print(
        "\n========== ENSEMBLE TEST RESULTS =========="
    )

    for k, v in ens_metrics.items():
        print(f"{k}: {v}")

    # ========================
    # 12. 保存预测结果
    # ========================
    result_df = df_test.reset_index(drop=True).copy()

    result_df["True_Activity"] = test_true
    result_df["Pred_Activity"] = ens_raw
    result_df["Pred_Log_Activity"] = ens_test_pred_log

    result_df["Abs_Error"] = np.abs(
        result_df["True_Activity"]
        - result_df["Pred_Activity"]
    )

    result_df["Relative_Error"] = (
        result_df["Abs_Error"]
        / np.maximum(result_df["True_Activity"], 1e-8)
    )

    result_df = result_df.sort_values(
        "Pred_Activity", ascending=False
    )

    result_df.to_csv(
        os.path.join(
            model_dir, "ensemble_predictions.csv"
        ),
        index=False
    )

    result_df.head(100).to_csv(
        os.path.join(
            model_dir, "ensemble_top100.csv"
        ),
        index=False
    )

    # ---------- 保存不同 seed 预测 ----------
    seed_df = df_test.reset_index(drop=True).copy()

    seed_df["True_Activity"] = test_true

    for i, seed in enumerate(seeds):

        seed_df[f"seed_{seed}"] = test_raw_preds[i]

    seed_df["Ensemble_Mean"] = ens_raw

    seed_df.to_csv(
        os.path.join(
            model_dir, "ensemble_seed_predictions.csv"
        ),
        index=False
    )

    # ---------- 保存指标 ----------
    pd.DataFrame([
        {"variant": "ensemble_raw", **ens_metrics}
    ]).to_csv(
        os.path.join(
            model_dir, "ensemble_metrics.csv"
        ),
        index=False
    )

    print("\n多 seed 集成预测结果已保存！")


if __name__ == "__main__":
    main()
