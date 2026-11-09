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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr


# ========================
# 1️⃣ 固定随机种子
# ========================
def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


# ========================
# 2️⃣ 标签变换
# ========================
def transform_activity(y):
    y = np.asarray(y, dtype=np.float32)
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_activity(y):
    y = np.asarray(y)
    return np.sign(y) * np.expm1(np.abs(y))


# ========================
# 3️⃣ Dataset 定义
# ========================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self, df, esm_embeds, mol_embeds1, mol_embeds2):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["Name"].astype(str).tolist()
        self.smiles1 = self.df["SM_胺"].tolist()
        self.smiles2 = self.df["SM_酮"].tolist()
        self.labels = pd.to_numeric(self.df["Activity"], errors="coerce").fillna(0).astype(float).to_numpy()
        self.labels_trans = transform_activity(self.labels)
        self.esm_embeds = esm_embeds; self.mol_embeds1 = mol_embeds1; self.mol_embeds2 = mol_embeds2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        esm_data = self.esm_embeds[self.idx[i]]
        esm_emb = torch.as_tensor(esm_data["emb"], dtype=torch.float32)
        esm_mask = torch.as_tensor(esm_data["mask"], dtype=torch.long)
        mol_feat1 = torch.as_tensor(self.mol_embeds1[self.smiles1[i]], dtype=torch.float32)
        mol_feat2 = torch.as_tensor(self.mol_embeds2[self.smiles2[i]], dtype=torch.float32)
        return esm_emb, esm_mask, mol_feat1, mol_feat2, torch.tensor(self.labels_trans[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32)


# ========================
# 4️⃣ 模型定义
# ========================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.25, hidden_dim=256):
        super().__init__()

        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj1 = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj2 = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.fc1 = nn.Linear(768, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)

        # 纯回归头
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, esm_feat, esm_mask, mol_feat1, mol_feat2):
        esm_feat = self.esm_proj(esm_feat)
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=(esm_mask == 0))
        mask = esm_mask.unsqueeze(-1).float()
        esm_pooled = (esm_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        mol1 = self.mol_proj1(mol_feat1)
        mol2 = self.mol_proj2(mol_feat2)
        x = torch.cat([esm_pooled, mol1, mol2], dim=-1)

        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))

        return self.reg_head(x).squeeze(-1)


# ========================
# 5️⃣ 评估函数
# ========================
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_pred, all_true, all_log_pred = [], [], []
    total_loss, total_num = 0, 0

    with torch.no_grad():
        for esm, mask, mol1, mol2, y, raw_y in loader:
            esm, mask, mol1, mol2, y = esm.to(device), mask.to(device), mol1.to(device), mol2.to(device), y.to(device)
            output = model(esm, mask, mol1, mol2)

            if criterion is not None:
                loss = criterion(output, y)
                total_loss += loss.item() * len(raw_y)
                total_num += len(raw_y)

            pred = inverse_activity(output.cpu().numpy())
            all_pred.extend(pred.tolist())
            all_true.extend(raw_y.numpy().tolist())
            all_log_pred.extend(output.cpu().numpy().tolist())

    y_true, y_pred = np.array(all_true), np.array(all_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    pearson = pearsonr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 and len(y_true) > 1 else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if len(np.unique(y_true)) > 1 else np.nan

    # 非零活性样本排序
    active = y_true != 0
    active_spearman = spearmanr(y_true[active], y_pred[active]).statistic if active.sum() > 1 and len(np.unique(y_true[active])) > 1 else np.nan
    active_mae = mean_absolute_error(y_true[active], y_pred[active]) if active.sum() else np.nan

    # Top100 排序评价
    k = min(100, len(y_true))
    true_top = np.argsort(-y_true)[:k]
    pred_top = np.argsort(-y_pred)[:k]
    overlap = len(set(true_top) & set(pred_top)) / k if k else np.nan

    # NDCG要求真实相关性非负，因此整体平移标签，不改变标签排序
    relevance = y_true - min(0, np.min(y_true))
    ndcg = ndcg_score(relevance.reshape(1, -1), y_pred.reshape(1, -1), k=k) if k else np.nan

    metrics = {"Loss": total_loss / total_num if total_num else np.nan, "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pearson, "Spearman": spearman, "Active_MAE": active_mae, "Active_Spearman": active_spearman, "NDCG@100": ndcg, "Top100_Overlap": overlap, "Top100_True_Mean": np.mean(y_true[pred_top]) if k else np.nan}
    predictions = {"true": y_true, "pred": y_pred, "pred_log": np.array(all_log_pred)}
    return metrics, predictions


# ========================
# 6️⃣ 训练函数
# ========================
def train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=500, patience=50, model_dir=None):
    os.makedirs(model_dir, exist_ok=True)
    best_metric, patience_counter = -float("inf"), 0
    history = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for esm, mask, mol1, mol2, y, raw_y in train_loader:
            esm, mask, mol1, mol2, y = esm.to(device), mask.to(device), mol1.to(device), mol2.to(device), y.to(device)
            output = model(esm, mask, mol1, mol2)
            loss = criterion(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics, _ = evaluate(model, val_loader, device, criterion)
        val_ndcg = val_metrics["NDCG@100"]
        monitor = val_ndcg if np.isfinite(val_ndcg) else -val_metrics["Loss"]
        scheduler.step(monitor)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_metrics['Loss']:.4f} | MAE={val_metrics['MAE']:.4f} | RMSE={val_metrics['RMSE']:.4f} | R2={val_metrics['R2']:.4f} | Spearman={val_metrics['Spearman']:.4f} | Active Spearman={val_metrics['Active_Spearman']:.4f} | NDCG@100={val_ndcg:.4f} | Top100 Overlap={val_metrics['Top100_Overlap']:.4f}")

        history.append({"epoch": epoch+1, "train_loss": train_loss, **{"val_"+k:v for k,v in val_metrics.items()}})

        if monitor > best_metric:
            best_metric, patience_counter = monitor, 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("保存最佳模型！")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(os.path.join(model_dir, "training_history.csv"), index=False)
    print("训练结束，最佳验证指标：", best_metric)


# ========================
# 7️⃣ 主程序
# ========================
def main():
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_dir = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED"
    feature_dir = os.path.join(base_dir, "Feature")
    split_dir = os.path.join(base_dir, "split_data_Activaty")
    model_dir = os.path.join(base_dir, "model_activity_regression")
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 加载特征 ----------
    with open(os.path.join(feature_dir, "esm_feats_emb_mask.pkl"), "rb") as f:
        esm_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "SM_胺.pkl"), "rb") as f:
        mol_dict1 = pickle.load(f)
    with open(os.path.join(feature_dir, "SM_酮.pkl"), "rb") as f:
        mol_dict2 = pickle.load(f)

    print(f"ESM={len(esm_dict)}, MolFormer胺={len(mol_dict1)}, MolFormer酮={len(mol_dict2)}")

    # ---------- 读取已经划分好的三个txt ----------
    def read_txt(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_idx = set(read_txt(os.path.join(split_dir, "train.txt")))
    val_idx = set(read_txt(os.path.join(split_dir, "val.txt")))
    test_idx = set(read_txt(os.path.join(split_dir, "test.txt")))

    # ---------- 读取原始CSV并按txt划分 ----------
    df = pd.read_csv(os.path.join(base_dir, "train_Activity_std2.csv"))
    df["idx"] = df["idx"].astype(str).str.strip()

    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)
    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)
    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)

    print(f"Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print(f"Train Activity分布:\n{df_train['Activity'].describe()}")
    print(f"Val Activity分布:\n{df_val['Activity'].describe()}")
    print(f"Test Activity分布:\n{df_test['Activity'].describe()}")

    if min(len(df_train), len(df_val), len(df_test)) == 0:
        raise ValueError("训练集、验证集或测试集为空，请检查txt中的idx是否与CSV的idx一致！")

    # ---------- Dataset ----------
    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict1, mol_dict2)
    val_set = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict1, mol_dict2)
    test_set = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict1, mol_dict2)

    batch_size = 256
    drop_last = len(train_set) % batch_size == 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 模型 ----------
    model = EnzymeSubstrateRegressor(esm_dim=1280, mol_dim=768, dropout=0.25, hidden_dim=256).to(device)
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # ---------- 训练 ----------
    train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=500, patience=50, model_dir=model_dir)

    # ---------- 测试 ----------
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device))
    test_metrics, test_pred = evaluate(model, test_loader, device, criterion)

    print("\n========== TEST RESULTS ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # ---------- 保存预测 ----------
    result_df = df_test.reset_index(drop=True).copy()
    result_df["True_Activity"] = test_pred["true"]
    result_df["Pred_Activity"] = test_pred["pred"]
    result_df["Pred_Log_Activity"] = test_pred["pred_log"]
    result_df["Abs_Error"] = np.abs(result_df["True_Activity"] - result_df["Pred_Activity"])
    result_df = result_df.sort_values("Pred_Activity", ascending=False)

    result_df.to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)
    result_df.head(100).to_csv(os.path.join(model_dir, "test_top100.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(model_dir, "test_metrics.csv"), index=False)

    print("\n测试集预测结果已保存！")


if __name__ == "__main__":
    main()
