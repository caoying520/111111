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
    def __init__(self, df, esm_embeds, mol_embeds):
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
        
        self.smiles = df["SM"].tolist()
        if "Conv" in df.columns:
            self.labels = df["Conv"].astype(float).tolist()
        else:
            self.labels = [0.0] * len(df)
        self.labels = df["Conv"].astype(float).tolist()
        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        # 归一化后浓度列（log+zscore，在 main() 中计算）
        if "SM_norm" not in df.columns or "Enz_norm" not in df.columns:
            raise ValueError("DataFrame must contain 'SM_norm' and 'Enz_norm' columns.")
        self.sm_norm = df["SM_norm"].astype(float).tolist()
        self.enzyme_norm = df["Enz_norm"].astype(float).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):   #CSV 每一行的 Name 能在 ESM.pkl 找到；这一行的 SM 能在 SM.pkl 找到（通过Name联系）  。
        idx = self.idx[i]     #第几行的Name
        esm_data = self.esm_embeds[idx]  # {"emb": tensor, "mask": tensor}
        esm_emb = esm_data["emb"].float()    # e.g. [L, d]                    去ESM.pkl 中找到对应的emb特征，相同序列和name对应的emb是一样的
        esm_mask = esm_data["mask"].long()   # e.g. [L]                       去ESM.pkl 中找到对应的mask特征
        mol_feat = torch.tensor(self.mol_embeds[self.smiles[i]], dtype=torch.float32) #self.smiles[i]中得到SM序列，再去SM.pkl 中找到对应的 768维MolFormer特征

        # 浓度：归一化后的标量，保持 shape [1]
        sm_load = torch.tensor([self.sm_norm[i]], dtype=torch.float32)
        enz_load = torch.tensor([self.enzyme_norm[i]], dtype=torch.float32)

        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return esm_emb, esm_mask, mol_feat, sm_load, enz_load, label


# # ========================
# # 3️⃣ 模型定义（包含两个浓度的 1->128 映射）
# # ========================
# class EnzymeSubstrateRegressor(nn.Module):
#     def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
#         super().__init__()
#         # ESM projection: esm_dim -> 256 (via 512 -> 256)
#         self.esm_proj = nn.Sequential(
#             nn.Linear(esm_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#         )
#         # Mol projection: mol_dim -> 256
#         self.mol_proj = nn.Sequential(
#             nn.Linear(mol_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#         )

#         # Transformer encoder for esm sequence tokens (keeps d_model=256)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True
#         )
#         #self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.esm_encoder = nn.TransformerEncoder(encoder_layer,num_layers=2,enable_nested_tensor=False)

#         # 浓度投影：1 -> 64 -> 128（分别为 SM 和 Enzyme）
#         self.sm_proj = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.LayerNorm(128)
#         )
#         self.enz_proj = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.LayerNorm(128)
#         )

#         # 全连接层：输入维度 256 + 256 + 128 + 128 = 768
#         input_dim = 256 + 256 + 128 + 128
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
#         self.fc4 = nn.Linear(hidden_dim // 2, 1)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, esm_feat, esm_mask, mol_feat, sm_load, enz_load):
#         # esm_feat: [B, L, esm_dim]  (if loaded as [L, d] per sample, DataLoader must batch consistently)
#         # esm_mask: [B, L]
#         # mol_feat: [B, mol_dim]
#         # sm_load, enz_load: [B, 1]

#         # Project token features
#         esm_feat = self.esm_proj(esm_feat)  # [B, L, 256]

#         # Build src_key_padding_mask for Transformer: True where padding (mask==0)
#         src_mask = esm_mask == 0  # [B, L] bool
#         src_mask = src_mask.bool()
      
#         # Transformer Encoder     
#         esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask)  # [B, L, 256]
#         #print("esm_mask:", esm_mask.shape)
        
#         # #新增
#         # actual_seq_len = esm_feat.size(1)
#         # # 同步裁剪 mask，确保它的长度和 esm_feat 的第二维完全一致.这样无论 PyTorch 怎么缩减长度，相乘时都不会报错
#         # esm_mask = esm_mask[:, :actual_seq_len]

#         # Masked mean pooling over sequence length
#         mask_expanded = esm_mask.unsqueeze(-1).float()  # [B, L, 1]
#         denom = mask_expanded.sum(dim=1) + 1e-8  # [B, 1]
#         esm_pooled = (esm_feat * mask_expanded).sum(dim=1) / denom  # [B, 256]

#         # Mol projection
#         mol_proj = self.mol_proj(mol_feat)  # [B, 256]

#         # Concentration projections
#         sm_proj = self.sm_proj(sm_load)   # [B, 128]
#         enz_proj = self.enz_proj(enz_load)  # [B, 128]

#         # Concat all
#         x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj], dim=-1)  # [B, 768]

#         # FC blocks with residual
#         x = F.relu(self.bn1(self.fc1(x)))
#         residual = x
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = x + residual
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = self.fc4(x)  # [B, 1]
#         return x.squeeze(-1)  # [B]


class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()
        # ESM projection: esm_dim:1280 -> 256 (via 512 -> 256)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        # Mol projection: mol_dim:768 -> 256
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

        # Transformer encoder for esm sequence tokens (keeps d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        #self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.esm_encoder = nn.TransformerEncoder(encoder_layer,num_layers=2,enable_nested_tensor=False)

        # 浓度投影：1 -> 64 -> 128（分别为 SM 和 Enzyme）
        self.sm_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        self.enz_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        # # 全连接层：输入维度 256 + 256 + 128 + 128 = 768
        # input_dim = 256 + 256 + 128 + 128
        # 256 + 256 + 256 + 256 + 128 + 128 = 1280
        input_dim = 1280
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, esm_feat, esm_mask, mol_feat, sm_load, enz_load):
        # esm_feat: [B, L, esm_dim]  (if loaded as [L, d] per sample, DataLoader must batch consistently)
        # esm_mask: [B, L]
        # mol_feat: [B, mol_dim]
        # sm_load, enz_load: [B, 1]

        # Project token features
        esm_feat = self.esm_proj(esm_feat)  # [B, L, 256]

        # Build src_key_padding_mask for Transformer: True where padding (mask==0)
        src_mask = esm_mask == 0  # [B, L] bool
        # src_mask = src_mask.bool()
              
        # Transformer Encoder     
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask)  # [B, L, 256]
        #print("esm_mask:", esm_mask.shape)
        
        # #新增
        # actual_seq_len = esm_feat.size(1)
        # # 同步裁剪 mask，确保它的长度和 esm_feat 的第二维完全一致.这样无论 PyTorch 怎么缩减长度，相乘时都不会报错
        # esm_mask = esm_mask[:, :actual_seq_len]   #多余padding部分是0也可以删掉

        # Masked mean pooling over sequence length
        mask_expanded = esm_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask_expanded.sum(dim=1) + 1e-8  # [B, 1]
        esm_pooled = (esm_feat * mask_expanded).sum(dim=1) / denom  # [B, 256]

        # Mol projection
        mol_proj = self.mol_proj(mol_feat)  # [B, 256]

        # Concentration projections
        sm_proj = self.sm_proj(sm_load)   # [B, 128]
        enz_proj = self.enz_proj(enz_load)  # [B, 128]

        # Concat all
        # x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj], dim=-1)  # [B, 768]
        
        # ===== 显式 Enzyme-Substrate interaction =====
        interaction_mul = esm_pooled * mol_proj
        interaction_abs = torch.abs(esm_pooled - mol_proj)

        # ===== Fusion =====
        x = torch.cat([
            esm_pooled,
            mol_proj,
            interaction_mul,
            interaction_abs,
            sm_proj,
            enz_proj
        ], dim=-1)

        # FC blocks with residual
        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # [B, 1]
        return x.squeeze(-1)  # [B]

# ========================
# 4️⃣ 评估函数（与原脚本保持一致）
# ========================
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in loader:
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)
            y = y.to(device)

            outputs = model(esm_feat, esm_mask, mol_feat, sm_load, enz_load)
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())
            if criterion is not None:
                total_loss += criterion(outputs, y).item()

    preds = np.array(preds)
    trues = np.array(trues)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    pearson, _ = pearsonr(trues, preds)
    avg_loss = total_loss / len(loader) if criterion is not None else None
    return avg_loss, mse, rmse, mae, r2, pearson

# ========================
# 5️⃣ 训练函数（保存 top-5 + best）
# ========================
def train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=2000):
    best_r2 = -np.inf
    top_models = []  # min-heap of (r2, path)

    model_dir = "model_resnet_krd_conc_S2_KRD_M_55%_add_random"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        # for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in loop:
        for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in train_loader:
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)
            y = y.to(device)

            preds = model(esm_feat, esm_mask, mol_feat, sm_load, enz_load)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate after epoch
        train_loss, train_mse, train_rmse, train_mae, train_r2, train_p = evaluate(model, train_loader, device, criterion)
        val_loss, val_mse, val_rmse, val_mae, val_r2, val_p = evaluate(model, val_loader, device, criterion)

        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Train: Loss={train_loss:.4f} | MSE={train_mse:.4f} | RMSE={train_rmse:.4f} | MAE={train_mae:.4f} | "
            f"R2={train_r2:.4f} | Pearson={train_p:.4f} || "
            f"Val: Loss={val_loss:.4f} | MSE={val_mse:.4f} | RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | "
            f"R2={val_r2:.4f} | Pearson={val_p:.4f}"
        )

        # Save checkpoint for this epoch if top-5 by val_r2
        model_path = os.path.join(model_dir, f"epoch_{epoch+1:04d}_r2_{val_r2:.4f}.pth")
        if len(top_models) < 5:
            heapq.heappush(top_models, (val_r2, model_path))
            torch.save(model.state_dict(), model_path)
        elif val_r2 > top_models[0][0]:
            worst_r2, worst_path = heapq.heappop(top_models)
            heapq.heappush(top_models, (val_r2, model_path))
            torch.save(model.state_dict(), model_path)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

        # Save best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

    print("\nTop-5 validation models saved:")
    for r2, path in sorted(top_models, key=lambda x: -x[0]):
        print(f"  R2={r2:.4f} | {path}")

# ========================
# 6️⃣ 主程序
# ========================
def main():
    set_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------- 加载特征（ESM & Mol） ----------
    # esm_pkl = "/home/qian.xu/enzyme/embeddings/esm_feats_emb_mask_merged_KRD_S2_KRD_M.pkl"
    # mol_pkl = "/home/qian.xu/enzyme/embeddings/mol_feats_merged_KRD_S2_KRD_M.pkl"
    esm_pkl = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_trian/esm_feats_emb_mask_train.pkl"
    mol_pkl = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_trian/SM_random.pkl"
    with open(esm_pkl, "rb") as f:
        esm_dict = pickle.load(f)
    with open(mol_pkl, "rb") as f:
        mol_dict = pickle.load(f)
    print(f"✅ 已加载 ESM 特征: {len(esm_dict)} 条 | MolFormer 特征: {len(mol_dict)} 条")

    # ---------- 数据划分（使用你已有的 split txt） ----------
    split_dir = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/data/酶筛/splits/add"
    with open(os.path.join(split_dir, "train_S2_KRD_M_50%_add.txt")) as f:
        train_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "val_S2_KRD_M_50%_5%_add.txt")) as f:
        val_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "test_add.txt")) as f:
        test_idx = [line.strip() for line in f if line.strip()]
    # ---------- 数据读取 ----------
    # csv_path = "/home/qian.xu/enzyme/data/out/merged_KRD_S2_KRD_M.csv"
    csv_path = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/data/酶筛/merged_KRD_S2_KRD_M_add.csv"
    df = pd.read_csv(csv_path)
    print("Loaded df:", df.shape)
    df["idx"] = df["idx"].astype(str)
    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)
    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)
    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)
    print(f"📂 数据集大小: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    SM_COL = "SM loading (g/L)"
    ENZ_COL = "Enzyme loading (g/L)"

    for d in [df_train, df_val, df_test]:
        d[SM_COL] = pd.to_numeric(d[SM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        d[ENZ_COL] = pd.to_numeric(d[ENZ_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        d["SM_log"] = np.log1p(d[SM_COL].astype(float))
        d["Enz_log"] = np.log1p(d[ENZ_COL].astype(float))

    sm_mean = df_train["SM_log"].mean()
    sm_std = df_train["SM_log"].std() + 1e-12
    enz_mean = df_train["Enz_log"].mean()
    enz_std = df_train["Enz_log"].std() + 1e-12

    for d in [df_train, df_val, df_test]:
        d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
        d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std

    print("SM_log mean/std (train):", sm_mean, sm_std)
    print("Enz_log mean/std (train):", enz_mean, enz_std)

    # ---------- Dataset & DataLoader ----------
    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict)
    val_set = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict)
    test_set = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 模型、优化器、损失 ----------
    model = EnzymeSubstrateRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # ---------- 训练 ----------
    train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=2000)

    # ---------- 测试评估 ----------
    best_model_path = "model_resnet_krd_conc_S2_KRD_M_55%_add_random/best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_loss, mse, rmse, mae, r2, pearson = evaluate(model, test_loader, device, criterion)
    print(f"\n🧪 Test | Loss={test_loss:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | "
          f"MAE={mae:.4f} | R2={r2:.4f} | Pearson={pearson:.4f}")

if __name__ == "__main__":
    main()
