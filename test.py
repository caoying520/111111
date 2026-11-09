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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import heapq
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import pearsonr, spearmanr
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
    def __init__(self, df, esm_embeds, mol_embeds1,mol_embeds2,threshold=0.03):
        """
        df: pandas.DataFrame, must include columns:
            - name (key to esm_embeds)
            - SMILES (key to mol_embeds)
            - Conv (label)
            - SM_norm (归一化后的底物浓度)
            - Enz_norm (归一化后的酶浓度)
        esm_embeds: dict[name] -> {"emb": tensor[L, d], "mask": tensor[L]}
        mol_embeds: dict[smiles] -> vector (numpy/torch)
        """
        self.df = df.reset_index(drop=True)
        self.idx = df["Name"].astype(str).tolist()
        self.smiles1 = df["SM_胺"].tolist()
        self.smiles2 = df["SM_酮"].tolist()
        # self.labels = df["Activity"].astype(float).tolist()
        self.labels = pd.to_numeric(self.df['Activity'],errors="coerce").fillna(0).astype(float).to_numpy()
        n_neg_label = int((self.labels < 0).sum())
        if n_neg_label:
            print(f"⚠️ 标签中有 {n_neg_label} 条负值 Activity，已截断为 0（按无活性处理）")
        self.labels =np.clip(self.labels,0.0,None)
        self.esm_embeds = esm_embeds
        self.mol_embeds1 = mol_embeds1
        self.mol_embeds2 = mol_embeds2
        self.threshold = threshold

        # # 归一化后浓度列（log+zscore，在 main() 中计算）
        # if "SM_norm" not in df.columns or "Enz_norm" not in df.columns:
        #     raise ValueError("DataFrame must contain 'SM_norm' and 'Enz_norm' columns.")
        # self.sm_norm = df["SM_norm"].astype(float).tolist()
        # self.enzyme_norm = df["Enz_norm"].astype(float).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.idx[i]
        esm_data = self.esm_embeds[idx]  # {"emb": tensor, "mask": tensor}
        esm_emb = esm_data["emb"].float()    # e.g. [L, d]
        esm_mask = esm_data["mask"].long()   # e.g. [L]
        mol_feat1 = torch.tensor(self.mol_embeds1[self.smiles1[i]], dtype=torch.float32)
        mol_feat2 = torch.tensor(self.mol_embeds2[self.smiles2[i]], dtype=torch.float32)

        activity = float(self.labels[i])
        #分类标签
        cls_label = float(activity >= self.threshold)
        #回归标签
        reg_label = float(np.log1p(activity))  # 浓度：归一化后的标量，保持 shape [1]
        #仅有效活性样本参与回归
        reg_mask =float(activity >= self.threshold)  
        
        # sm_load = torch.tensor([self.sm_norm[i]], dtype=torch.float32)
        # enz_load = torch.tensor([self.enzyme_norm[i]], dtype=torch.float32)

        # label = torch.tensor(self.labels[i], dtype=torch.float32)
        return esm_emb, esm_mask, mol_feat1, mol_feat2, torch.tensor(cls_label,dtype=torch.float32), torch.tensor(reg_label,dtype=torch.float32), torch.tensor(reg_mask),torch.tensor(activity,dtype=torch.float32)

# ========================
# 3️⃣ 模型定义（包含两个浓度的 1->128 映射）
# ========================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()
        # ESM projection: esm_dim -> 256 (via 512 -> 256)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        # Mol projection: mol_dim -> 256
        self.mol_proj1 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        self.mol_proj2 = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        # Transformer encoder for esm sequence tokens (keeps d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        # enable_nested_tensor=False：关闭嵌套张量快路径，
        # 否则带 padding mask 时输出长度会被截到"本 batch 最长有效序列"（如 325），
        # 与固定长度 400 的 mask 对不上，masked-mean-pooling 会报尺寸不匹配
        self.esm_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )

        # # 浓度投影：1 -> 64 -> 128（分别为 SM 和 Enzyme）
        # self.sm_proj = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.LayerNorm(128)
        # )
        # self.enz_proj = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.LayerNorm(128)
        # )

        # 全连接层：输入维度 256 + 256 + 128 + 128 = 768
        # input_dim = 256 + 256 + 128 + 128
        input_dim = 256 + 256 + 256
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        #分类头
        self.cls_head = nn.Linear(hidden_dim // 2, 1)
        #回归头
        self.reg_head = nn.Linear(hidden_dim // 2, 1)
        # self.fc4 = nn.Linear(hidden_dim // 2, 1)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, esm_feat, esm_mask, mol_feat1, mol_feat2):
        # esm_feat: [B, L, esm_dim]  (if loaded as [L, d] per sample, DataLoader must batch consistently)
        # esm_mask: [B, L]
        # mol_feat: [B, mol_dim]
        # sm_load, enz_load: [B, 1]

        # Project token features
        esm_feat = self.esm_proj(esm_feat)  # [B, L, 256]

        # Build src_key_padding_mask for Transformer: True where padding (mask==0)
        src_mask = esm_mask == 0  # [B, L] bool

        # Transformer Encoder
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask)  # [B, L, 256]

        # Masked mean pooling over sequence length
        mask_expanded = esm_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask_expanded.sum(dim=1) + 1e-8  # [B, 1]
        esm_pooled = (esm_feat * mask_expanded).sum(dim=1) / denom  # [B, 256]

        # Mol projection
        mol_proj1 = self.mol_proj1(mol_feat1)  # [B, 256]
        mol_proj2 = self.mol_proj2(mol_feat2)  # [B, 256]

        # # Concentration projections
        # sm_proj = self.sm_proj(sm_load)   # [B, 128]
        # enz_proj = self.enz_proj(enz_load)  # [B, 128]

        # Concat all
        # x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj], dim=-1)  # [B, 768]

        x = torch.cat([esm_pooled, mol_proj1, mol_proj2], dim=-1)  # [B, 768 + 256 + 256]

        # FC blocks with residual
        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual
        x = F.relu(self.bn3(self.fc3(x)))

        x=self.dropout(x)
        cls_pred = self.cls_head(x).squeeze(-1)  # [B, 1]
        reg_pred = self.reg_head(x).squeeze(-1)  # [B, 1]
        # x = self.fc4(x)  # [B, 1]
        # return x.squeeze(-1)  # [B]
        return cls_pred, reg_pred

# 多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=4.0,lambda_cls=1.0,lambda_reg=1.0):

        super().__init__()

        self.register_buffer("pos_weight",torch.tensor(float(pos_weight)))
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.reg_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, cls_logits, reg_output,
                cls_target, reg_target, reg_mask):

        # 分类损失：全部样本
        loss_cls = self.cls_loss(cls_logits, cls_target)
        # 回归损失：仅有效活性样本
        loss_reg_each = self.reg_loss(reg_output, reg_target)
        loss_reg = (loss_reg_each * reg_mask).sum() / reg_mask.sum().clamp_min(1.0)

        loss = (self.lambda_cls * loss_cls+ self.lambda_reg * loss_reg)

        return loss, loss_cls, loss_reg


# ========================
# 4️⃣ 评估函数（与原脚本保持一致）
# ========================
def evaluate(model, loader, device, criterion=None,threshold=0.03):
    model.eval()
    all_cls, all_reg = [], []
    all_cls_y, all_reg_y = [], []
    all_raw_y = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            (esm_feat, esm_mask, mol_feat1, mol_feat2,cls_y, reg_y, reg_mask, raw_y) = batch
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)
            cls_y = cls_y.to(device)
            reg_y = reg_y.to(device)
            reg_mask = reg_mask.to(device)

            cls_logits, reg_output = model(esm_feat, esm_mask, mol_feat1, mol_feat2)
            if criterion is not None:
                loss, _, _ = criterion(cls_logits, reg_output,cls_y, reg_y, reg_mask)
                total_loss += loss.item() * len(raw_y)
                total_samples += len(raw_y)

            cls_prob = torch.sigmoid(cls_logits)

            # log1p(Activity) 转换回原始活性尺度
            reg_pred = torch.expm1(reg_output.clamp(min=0, max=10))
            # 期望活性估计
            expected_activity = cls_prob * reg_pred

            all_cls.extend(cls_prob.cpu().numpy())
            all_reg.extend(reg_pred.cpu().numpy())
            all_cls_y.extend(cls_y.cpu().numpy())
            all_reg_y.extend(reg_y.cpu().numpy())
            all_raw_y.extend(raw_y.cpu().numpy())

    all_cls = np.array(all_cls)
    all_reg = np.array(all_reg)
    all_raw_y = np.array(all_raw_y)

    all_cls_y = np.array(all_cls_y).astype(int)

    expected_activity = all_cls * all_reg

    # 分类指标
    pred_class = (all_cls >= 0.5).astype(int)

    acc = accuracy_score(all_cls_y, pred_class)
    precision = precision_score(all_cls_y, pred_class, zero_division=0)
    recall = recall_score(all_cls_y, pred_class, zero_division=0)
    f1 = f1_score(all_cls_y, pred_class, zero_division=0)

    if len(np.unique(all_cls_y)) == 2:
        auc = roc_auc_score(all_cls_y, all_cls)
        pr_auc = average_precision_score(all_cls_y, all_cls)
    else:
        auc, pr_auc = np.nan, np.nan

    # 原始活性尺度下的整体预测
    mse = mean_squared_error(all_raw_y, expected_activity)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_raw_y, expected_activity)
    r2 = r2_score(all_raw_y, expected_activity)

    # 只评价真实有效活性样本的条件回归能力
    active_mask = all_raw_y >= threshold

    if active_mask.sum() > 1:
        active_mae = mean_absolute_error(all_raw_y[active_mask],all_reg[active_mask])
        active_rmse = np.sqrt(mean_squared_error(all_raw_y[active_mask],all_reg[active_mask]))
        active_r2 = r2_score(all_raw_y[active_mask],all_reg[active_mask])
        active_spearman = spearmanr(all_raw_y[active_mask],all_reg[active_mask]).statistic
    else:
        active_mae = np.nan
        active_rmse = np.nan
        active_r2 = np.nan
        active_spearman = np.nan
    avg_loss = (total_loss / total_samples if criterion is not None else None)

    metrics = {
        "Loss": avg_loss,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Active_MAE": active_mae,
        "Active_RMSE": active_rmse,
        "Active_R2": active_r2,
        "Active_Spearman": active_spearman
    }

    predictions = {
        "true": all_raw_y,
        "prob": all_cls,
        "conditional_activity": all_reg,
        "expected_activity": expected_activity
    }

    return metrics, predictions


# ========================
# 5️⃣ 训练函数（保存 top-5 + best）
# ========================
def train_model(train_loader,val_loader,model,optimizer,criterion,device,epochs=300,patience=30):

    model_dir = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/model_activity_Activity"
    os.makedirs(model_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    history = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0

        for batch in train_loader:
            (esm_feat, esm_mask, mol_feat1, mol_feat2,cls_y, reg_y, reg_mask, raw_y) = batch
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)

            cls_y = cls_y.to(device)
            reg_y = reg_y.to(device)
            reg_mask = reg_mask.to(device)

            cls_logits, reg_output = model(esm_feat, esm_mask, mol_feat1, mol_feat2)

            loss, loss_cls, loss_reg = criterion(cls_logits,reg_output,cls_y,reg_y,reg_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_reg += loss_reg.item()

        n_batch = len(train_loader)
        train_loss = total_loss / n_batch
        train_cls = total_cls / n_batch
        train_reg = total_reg / n_batch

        # 验证
        val_metrics, val_pred = evaluate(model, val_loader, device, criterion)
        val_loss = val_metrics["Loss"]
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} (cls={train_cls:.4f}, reg={train_reg:.4f}) | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['F1']:.4f} | "
            f"Val PR-AUC={val_metrics['PR_AUC']:.4f} | "
            f"Val MAE={val_metrics['MAE']:.4f} | "
            f"Active MAE={val_metrics['Active_MAE']:.4f} | "
            f"Active R2={val_metrics['Active_R2']:.4f}"
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
            torch.save(model.state_dict(),os.path.join(model_dir, "best_model.pth"))
            print("保存最佳模型！")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early Stopping at Epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(os.path.join(model_dir, "training_history.csv"),index=False)

    print(f"\n训练结束，最佳验证损失：{best_val_loss:.6f}")

# ========================
# 6️⃣ 主程序
# ========================
def main():
    set_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------- 加载特征（ESM & Mol） ----------
    esm_pkl = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/Feature/esm_feats_emb_mask.pkl"
    mol_pkl1 = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/Feature/SM_胺.pkl"
    mol_pkl2 = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/Feature/SM_酮.pkl"
    with open(esm_pkl, "rb") as f:
        esm_dict = pickle.load(f)
    with open(mol_pkl1, "rb") as f:
        mol_dict1 = pickle.load(f)
    with open(mol_pkl2, "rb") as f:
        mol_dict2 = pickle.load(f)
        
    print(f"✅ 已加载 ESM 特征: {len(esm_dict)} 条 | MolFormer 特征: {len(mol_dict1)} 条 | MolFormer 特征: {len(mol_dict2)} 条")

    # ---------- 数据划分（使用你已有的 split txt） ----------
    split_dir = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/split_data_Activaty"
    with open(os.path.join(split_dir, "train.txt")) as f:
        train_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "val.txt")) as f:
        val_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "test.txt")) as f:
        test_idx = [line.strip() for line in f if line.strip()]
    # ---------- 数据读取 ----------
    csv_path = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/train_Activity_std2.csv"
    df = pd.read_csv(csv_path)
    print("Loaded df:", df.shape)
    df["idx"] = df["idx"].astype(str)
    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)
    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)
    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)
    print(f"📂 数据集大小: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # SM_COL = "SM loading (g/L)"
    # ENZ_COL = "Enzyme loading (g/L)"

    # for d in [df_train, df_val, df_test]:
    #     d[SM_COL] = pd.to_numeric(d[SM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    #     d[ENZ_COL] = pd.to_numeric(d[ENZ_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        # d["SM_log"] = np.log1p(d[SM_COL].astype(float))
        # d["Enz_log"] = np.log1p(d[ENZ_COL].astype(float))

    # sm_mean = df_train["SM_log"].mean()
    # sm_std = df_train["SM_log"].std() + 1e-12
    # enz_mean = df_train["Enz_log"].mean()
    # enz_std = df_train["Enz_log"].std() + 1e-12

    # for d in [df_train, df_val, df_test]:
    #     d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
    #     d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std

    # print("SM_log mean/std (train):", sm_mean, sm_std)
    # print("Enz_log mean/std (train):", enz_mean, enz_std)

    # ---------- Dataset & DataLoader ----------
    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict1, mol_dict2, threshold=0.03)
    val_set = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict1, mol_dict2, threshold=0.03)
    test_set = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict1, mol_dict2, threshold=0.03)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 模型、优化器、损失 ----------
    model = EnzymeSubstrateRegressor(esm_dim=1280, mol_dim=768, dropout=0.25, hidden_dim=256).to(device)
    # Loss
    n_pos = (df_train["Activity"] >= 0.03).sum()
    n_neg = len(df_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    criterion = MultiTaskLoss(pos_weight=pos_weight,lambda_cls=1.0,lambda_reg=1.0).to(device)
    print("Positive weight:", pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # criterion = nn.MSELoss()

    # ---------- 训练 ----------
    train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=500,patience=50)

    # ---------- 测试评估 ----------
    best_model_path = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/model_activity_Activity/best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_metrics, test_predictions = evaluate(model,test_loader,device,criterion)
    print("\n========== TEST RESULTS ==========")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    # Save predictions
    result_df = df_test.reset_index(drop=True).copy()
    result_df["True_Activity"] = test_predictions["true"]
    result_df["Pred_Probability"] = test_predictions["prob"]
    result_df["Pred_Conditional_Activity"] = (test_predictions["conditional_activity"])
    result_df["Pred_Expected_Activity"] = (test_predictions["expected_activity"])

    result_df.to_csv("/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED/model_activity_Activity/test_predictions.csv",index=False)
    print("测试集预测结果已保存！")

    # test_loss, mse, rmse, mae, r2, pearson = evaluate(model, test_loader, device, criterion)
    # print(f"\n🧪 Test | Loss={test_loss:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | "
    #       f"MAE={mae:.4f} | R2={r2:.4f} | Pearson={pearson:.4f}")

if __name__ == "__main__":
    main()
