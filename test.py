# -*- coding: utf-8 -*-
#np.set_printoptions(threshold=np.inf)
#!/usr/bin/env python3
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import ndcg_score

# ========================
# 1️⃣ 固定随机种子
# ========================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ========================
# 2️⃣ Dataset 定义
# ========================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self, df, esm_embeds, mol_embeds1, mol_embeds2):
        self.df = df.reset_index(drop=True)

        self.idx = self.df["Name"].astype(str).tolist()
        self.smiles1 = self.df["SM_胺"].tolist()
        self.smiles2 = self.df["SM_酮"].tolist()

        self.labels = pd.to_numeric(
            self.df["Activity"], errors="coerce"
        ).fillna(0).astype(float).to_numpy()

        n_neg_label = int((self.labels < 0).sum())
        if n_neg_label:
            print(f"⚠️ 标签中有 {n_neg_label} 条负值 Activity，已截断为 0")
        self.labels = np.clip(self.labels, 0.0, None)

        self.esm_embeds = esm_embeds
        self.mol_embeds1 = mol_embeds1
        self.mol_embeds2 = mol_embeds2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.idx[i]

        esm_data = self.esm_embeds[idx]
        esm_emb = esm_data["emb"].float()
        esm_mask = esm_data["mask"].long()

        mol_feat1 = torch.tensor(
            self.mol_embeds1[self.smiles1[i]], dtype=torch.float32
        )
        mol_feat2 = torch.tensor(
            self.mol_embeds2[self.smiles2[i]], dtype=torch.float32
        )

        activity = float(self.labels[i])

        # 回归标签：log1p(Activity)，零值对应标签0
        reg_label = float(np.log1p(activity))

        return (
            esm_emb,
            esm_mask,
            mol_feat1,
            mol_feat2,
            torch.tensor(reg_label, dtype=torch.float32),
            torch.tensor(activity, dtype=torch.float32)
        )


# ========================
# 3️⃣ 模型定义
# ========================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768,
                 dropout=0.25, hidden_dim=256):
        super().__init__()

        # ESM projection
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        # Mol projection 1
        self.mol_proj1 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        # Mol projection 2
        self.mol_proj2 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        # Transformer encoder
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

        # 全连接层
        input_dim = 256 + 256 + 256

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)

        # 只有一个回归头
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, esm_feat, esm_mask, mol_feat1, mol_feat2):

        # ESM projection
        esm_feat = self.esm_proj(esm_feat)

        # Transformer mask
        src_mask = esm_mask == 0

        esm_feat = self.esm_encoder(
            esm_feat,
            src_key_padding_mask=src_mask
        )

        # Masked mean pooling
        mask_expanded = esm_mask.unsqueeze(-1).float()
        denom = mask_expanded.sum(dim=1) + 1e-8

        esm_pooled = (
            (esm_feat * mask_expanded).sum(dim=1) / denom
        )

        # Mol projection
        mol_proj1 = self.mol_proj1(mol_feat1)
        mol_proj2 = self.mol_proj2(mol_feat2)

        # Feature fusion
        x = torch.cat(
            [esm_pooled, mol_proj1, mol_proj2],
            dim=-1
        )

        # FC blocks with residual
        x = F.relu(self.bn1(self.fc1(x)))

        residual = x

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Regression output
        reg_pred = self.reg_head(x).squeeze(-1)

        return reg_pred


# ========================
# 4️⃣ 评估函数
# ========================
def evaluate(model, loader, device, criterion=None):

    model.eval()

    all_reg = []
    all_reg_y = []
    all_raw_y = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            (esm_feat, esm_mask, mol_feat1, mol_feat2,
             reg_y, raw_y) = batch

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)
            reg_y = reg_y.to(device)

            reg_output = model(
                esm_feat, esm_mask, mol_feat1, mol_feat2
            )

            if criterion is not None:
                loss = criterion(reg_output, reg_y)

                total_loss += loss.item() * len(raw_y)
                total_samples += len(raw_y)

            # log1p(Activity) 还原到原始活性尺度
            reg_pred = torch.expm1(
                reg_output.clamp(min=0, max=10)
            )

            all_reg.extend(reg_pred.cpu().numpy())
            all_reg_y.extend(reg_y.cpu().numpy())
            all_raw_y.extend(raw_y.cpu().numpy())

    all_reg = np.array(all_reg)
    all_raw_y = np.array(all_raw_y)

    # 整体回归指标
    mse = mean_squared_error(all_raw_y, all_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_raw_y, all_reg)
    r2 = r2_score(all_raw_y, all_reg)

    # 整体 Spearman 排序相关性
    if len(np.unique(all_raw_y)) > 1:
        overall_spearman = spearmanr(
            all_raw_y, all_reg
        ).statistic
    else:
        overall_spearman = np.nan

    # 只评价真实非零活性样本
    active_mask = all_raw_y > 0

    if active_mask.sum() > 1:

        active_mae = mean_absolute_error(
            all_raw_y[active_mask],
            all_reg[active_mask]
        )

        active_rmse = np.sqrt(
            mean_squared_error(
                all_raw_y[active_mask],
                all_reg[active_mask]
            )
        )

        active_r2 = r2_score(
            all_raw_y[active_mask],
            all_reg[active_mask]
        )

        active_spearman = spearmanr(
            all_raw_y[active_mask],
            all_reg[active_mask]
        ).statistic

    else:
        active_mae = np.nan
        active_rmse = np.nan
        active_r2 = np.nan
        active_spearman = np.nan

    # Top100 排序指标
    k = min(100, len(all_raw_y))

    true_top_idx = np.argsort(-all_raw_y)[:k]
    pred_top_idx = np.argsort(-all_reg)[:k]

    overlap = len(
        set(true_top_idx).intersection(set(pred_top_idx))
    ) / k

    # NDCG@100
    ndcg100 = ndcg_score(
        all_raw_y.reshape(1, -1),
        all_reg.reshape(1, -1),
        k=k
    )

    pred_top_activity = all_raw_y[pred_top_idx]

    top100_true_mean = np.mean(pred_top_activity)
    top100_true_median = np.median(pred_top_activity)

    avg_loss = (
        total_loss / total_samples
        if criterion is not None else None
    )

    metrics = {
        "Loss": avg_loss,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": overall_spearman,
        "Active_MAE": active_mae,
        "Active_RMSE": active_rmse,
        "Active_R2": active_r2,
        "Active_Spearman": active_spearman,
        "NDCG@100": ndcg100,
        "Top100_Overlap": overlap,
        "Top100_True_Mean": top100_true_mean,
        "Top100_True_Median": top100_true_median
    }

    predictions = {
        "true": all_raw_y,
        "pred_activity": all_reg
    }

    return metrics, predictions


# ========================
# 5️⃣ 训练函数
# ========================
def train_model(train_loader, val_loader, model,
                optimizer, criterion, device,
                epochs=500, patience=50):

    model_dir = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/model_activity_regression"
    )

    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            (esm_feat, esm_mask, mol_feat1, mol_feat2,
             reg_y, raw_y) = batch

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)
            reg_y = reg_y.to(device)

            reg_output = model(
                esm_feat, esm_mask, mol_feat1, mol_feat2
            )

            loss = criterion(reg_output, reg_y)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        # 验证
        val_metrics, val_pred = evaluate(
            model, val_loader, device, criterion
        )

        val_loss = val_metrics["Loss"]

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val MAE={val_metrics['MAE']:.4f} | "
            f"Val RMSE={val_metrics['RMSE']:.4f} | "
            f"Val R2={val_metrics['R2']:.4f} | "
            f"Val Spearman={val_metrics['Spearman']:.4f} | "
            f"Active Spearman={val_metrics['Active_Spearman']:.4f} | "
            f"NDCG@100={val_metrics['NDCG@100']:.4f} | "
            f"Top100 Overlap={val_metrics['Top100_Overlap']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{
                "val_" + k: v
                for k, v in val_metrics.items()
            }
        })

        # 保存最佳模型
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "best_model.pth")
            )

            print("保存最佳模型！")

        else:
            patience_counter += 1

        if patience_counter >= patience:

            print(f"Early Stopping at Epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(model_dir, "training_history.csv"),
        index=False
    )

    print(f"\n训练结束，最佳验证损失：{best_val_loss:.6f}")


# ========================
# 6️⃣ 主程序
# ========================
def main():

    set_seed(123)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------- 加载特征（ESM & Mol） ----------
    esm_pkl = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/Feature/esm_feats_emb_mask.pkl"
    )

    mol_pkl1 = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/Feature/SM_胺.pkl"
    )

    mol_pkl2 = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/Feature/SM_酮.pkl"
    )

    with open(esm_pkl, "rb") as f:
        esm_dict = pickle.load(f)

    with open(mol_pkl1, "rb") as f:
        mol_dict1 = pickle.load(f)

    with open(mol_pkl2, "rb") as f:
        mol_dict2 = pickle.load(f)

    print(
        f"✅ 已加载 ESM 特征: {len(esm_dict)} 条 | "
        f"MolFormer 特征: {len(mol_dict1)} 条 | "
        f"MolFormer 特征: {len(mol_dict2)} 条"
    )

    # ---------- 数据划分（使用已有的三个 txt） ----------
    split_dir = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/split_data_Activaty"
    )

    with open(os.path.join(split_dir, "train.txt")) as f:
        train_idx = [line.strip() for line in f if line.strip()]

    with open(os.path.join(split_dir, "val.txt")) as f:
        val_idx = [line.strip() for line in f if line.strip()]

    with open(os.path.join(split_dir, "test.txt")) as f:
        test_idx = [line.strip() for line in f if line.strip()]

    # ---------- 数据读取 ----------
    csv_path = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/train_Activity_std2.csv"
    )

    df = pd.read_csv(csv_path)

    print("Loaded df:", df.shape)

    df["idx"] = df["idx"].astype(str)

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
        f"📂 原始数据集大小: "
        f"train={len(df_train)}, "
        f"val={len(df_val)}, "
        f"test={len(df_test)}"
    )

    # ---------- 训练集零值抽样 ----------
    ZERO_KEEP_RATIO = 0.20

    df_train["Activity"] = pd.to_numeric(
        df_train["Activity"], errors="coerce"
    ).fillna(0.0)

    df_val["Activity"] = pd.to_numeric(
        df_val["Activity"], errors="coerce"
    ).fillna(0.0)

    df_test["Activity"] = pd.to_numeric(
        df_test["Activity"], errors="coerce"
    ).fillna(0.0)

    train_zero = df_train[
        df_train["Activity"] == 0
    ]

    train_nonzero = df_train[
        df_train["Activity"] != 0
    ]

    n_zero_keep = int(
        round(len(train_zero) * ZERO_KEEP_RATIO)
    )

    df_train_zero_sampled = train_zero.sample(
        n=n_zero_keep,
        random_state=123
    ) if n_zero_keep > 0 else train_zero.iloc[0:0]

    df_train = pd.concat(
        [train_nonzero, df_train_zero_sampled],
        axis=0
    ).sample(
        frac=1,
        random_state=123
    ).reset_index(drop=True)

    print("\n========== 训练集零值抽样结果 ==========")

    print(
        f"原始训练集: {len(train_nonzero) + len(train_zero)}"
    )

    print(f"全部非零样本: {len(train_nonzero)}")
    print(f"原始零值样本: {len(train_zero)}")
    print(f"保留零值样本: {len(df_train_zero_sampled)}")

    print(f"抽样后训练集: {len(df_train)}")

    print(
        f"抽样后零值比例: "
        f"{(df_train['Activity'] == 0).mean():.2%}"
    )

    # 保存实际参与训练的数据
    model_dir = (
        "/data/AIPD/R2/enzyme_交接20260609/"
        "enzyme_train/IRED/model_activity_regression"
    )

    os.makedirs(model_dir, exist_ok=True)

    df_train.to_csv(
        os.path.join(model_dir, "train_sampled.csv"),
        index=False
    )

    df_val.to_csv(
        os.path.join(model_dir, "val_full.csv"),
        index=False
    )

    df_test.to_csv(
        os.path.join(model_dir, "test_full.csv"),
        index=False
    )

    # ---------- Dataset & DataLoader ----------
    train_set = EnzymeSubstrateDataset(
        df_train, esm_dict, mol_dict1, mol_dict2
    )

    val_set = EnzymeSubstrateDataset(
        df_val, esm_dict, mol_dict1, mol_dict2
    )

    test_set = EnzymeSubstrateDataset(
        df_test, esm_dict, mol_dict1, mol_dict2
    )

    batch_size = 128

    # 避免 BatchNorm 遇到最后一个只有1条数据的 batch
    drop_last = (
        len(train_set) % batch_size == 1
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---------- 模型、优化器、损失 ----------
    model = EnzymeSubstrateRegressor(
        esm_dim=1280,
        mol_dim=768,
        dropout=0.25,
        hidden_dim=256
    ).to(device)

    # 单一回归损失
    criterion = nn.SmoothL1Loss(beta=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )

    # ---------- 训练 ----------
    train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion,
        device,
        epochs=500,
        patience=30
    )

    # ---------- 测试评估 ----------
    best_model_path = os.path.join(
        model_dir, "best_model.pth"
    )

    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    model.to(device)

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        device,
        criterion
    )

    print("\n========== TEST RESULTS ==========")

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    # ---------- 保存预测结果 ----------
    result_df = df_test.reset_index(drop=True).copy()

    result_df["True_Activity"] = test_predictions["true"]

    result_df["Pred_Activity"] = (
        test_predictions["pred_activity"]
    )

    result_df["Abs_Error"] = abs(
        result_df["True_Activity"] -
        result_df["Pred_Activity"]
    )

    result_df = result_df.sort_values(
        "Pred_Activity",
        ascending=False
    )

    result_df.to_csv(
        os.path.join(model_dir, "test_predictions.csv"),
        index=False
    )

    # 保存测试集预测 Top100
    result_df.head(100).to_csv(
        os.path.join(model_dir, "test_top100.csv"),
        index=False
    )

    # 保存测试集指标
    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        os.path.join(model_dir, "test_metrics.csv"),
        index=False
    )

    print("\n测试集预测结果已保存！")
    print("测试集 Top100 已保存！")


if __name__ == "__main__":
    main()
