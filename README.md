
    #                 原始模型
    #          Conv 0～1 回归模型
    #                    │
    #                    │ 加载 pretrained weights
    #                    ↓
    #    ┌────────────────────────────┐
    #    │ ESM-2                      │
    #    │ MolFormer                  │
    #    │ Transformer                │
    #    │ Enzyme-Substrate Fusion    │
    #    │ Concentration              │
    #    │ FC1 → FC2 → FC3            │
    #    └──────────────┬─────────────┘
    #                   │
    #              重新初始化（只适用于任务高度相关，前面的特征很好迁移（前面层冻结就行），反正往往需要解冻别的层）
    #                 FC4
    #                   │
    #                   ↓
    #           Activity 连续回归
    #                   │
    #        -0.2 ～ 0.0 ～ 0.08
    #                   │
    #             threshold=0.03
    #                   │
    #           ┌───────┴───────┐
    #           │               │
    #        < 0.03          ≥ 0.03
    #           │               │
    #           ↓               ↓
    #           0               1


# 原始模型权重
#       ↓
# 加载
#       ↓
# ESM + MolFormer + Transformer + Fusion + FC
#       ↓
# 用新 Activity 数据微调
#       ↓
# Activity 连续预测
#       ↓
# 0.03 阈值
#       ↓
# 0/1

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
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import pearsonr
import heapq
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 参数
# ============================================================

SEED = 123
BATCH_SIZE = 256

# 原始训练好的模型
PRETRAINED_MODEL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"

# 新 Activity 数据
CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/train_data3.csv"

# ESM / MolFormer
ESM_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/esm_feats_emb_mask_test_6_0.pkl"

MOL_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/SM.pkl"

# split
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/splits_Activity"

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"

# 输出
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_finetune"
os.makedirs(MODEL_DIR, exist_ok=True)

# 最终二分类阈值
ACTIVITY_THRESHOLD = 0.03


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
    def __init__(self, df, esm_embeds, mol_embeds,activity_mean,activity_std):
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

        activity = self.df["Activity"].astype(float).values

        #标准化Activity
        self.labels = ((activity - activity_mean) / activity_std)

        
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

        # 全连接层：输入维度 256 + 256 + 128 + 128 = 768
        input_dim = 256 + 256 + 128 + 128
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
        src_mask = src_mask.bool()
      
        # Transformer Encoder     
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask)  # [B, L, 256]
        #print("esm_mask:", esm_mask.shape)
        
        # #新增
        # actual_seq_len = esm_feat.size(1)
        # # 同步裁剪 mask，确保它的长度和 esm_feat 的第二维完全一致.这样无论 PyTorch 怎么缩减长度，相乘时都不会报错
        # esm_mask = esm_mask[:, :actual_seq_len]

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
        x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj], dim=-1)  # [B, 768]

        # FC blocks with residual
        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # [B, 1]
        return x.squeeze(-1)  # [B]

# ============================================================
# 5. 加载原始模型
#    除 fc4 外全部继承
# ============================================================

def load_pretrained(model, path, device):

    ckpt = torch.load(
        path,
        map_location=device
    )

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # 原 Conv 回归层不能直接用于 Activity
    ckpt.pop("fc4.weight", None)   #把原模型中的fc4删掉，fc4学的是具体标签，需重新训练，目的是让前面层这些特征不再对应转化率，而是活性（后续就是fc4是活性标签反向传播预测）
    ckpt.pop("fc4.bias", None)

    missing, unexpected = model.load_state_dict(ckpt,strict=False)  #加载除fc4的原模型参数。

    print("\n加载原模型：")
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    # 新 Activity 输出层重新初始化，转化率头重新初始化为活性预测头
    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)

    print("✅ 原模型特征层加载完成")
    print("✅ fc4 重新初始化用于 Activity")

    return model


# ============================================================
# 6. 预测
# ============================================================

def predict(model,loader,device,activity_mean,activity_std):

    model.eval()
    preds = []
    trues = []
    with torch.no_grad():

        for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in loader:

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)

            out = model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)

            preds.extend(out.cpu().numpy())

            trues.extend(y.numpy())

    preds = np.asarray(preds)
    trues = np.asarray(trues)

    # 恢复成真实 Activity
    preds = preds * activity_std + activity_mean
    trues = trues * activity_std + activity_mean

    return trues, preds

# ============================================================
# 7. 评估
# ============================================================

def evaluate_binary(true,pred,threshold=0.03):

    rmse = np.sqrt(mean_squared_error(true, pred))

    mae = mean_absolute_error(true, pred)

    r2 = r2_score(true, pred)

    pearson = pearsonr(true, pred)[0]

    true_bin = (true >= threshold).astype(int)

    pred_bin = (pred >= threshold).astype(int)

    acc = accuracy_score(
        true_bin, pred_bin
    )

    precision = precision_score(
        true_bin,
        pred_bin,
        zero_division=0
    )

    recall = recall_score(
        true_bin,
        pred_bin,
        zero_division=0
    )

    f1 = f1_score(
        true_bin,
        pred_bin,
        zero_division=0
    )

    if len(np.unique(true_bin)) == 2:
        auc = roc_auc_score(
            true_bin,
            pred
        )
    else:
        auc = np.nan

    cm = confusion_matrix(
        true_bin,
        pred_bin,
        labels=[0, 1]
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }


# ============================================================
# 8. 单个 epoch
# ============================================================

def train_epoch(model,loader,optimizer,criterion,device):

    model.train()

    total_loss = 0

    for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in loader:

        esm_feat = esm_feat.to(device)
        esm_mask = esm_mask.to(device)
        mol_feat = mol_feat.to(device)
        sm_load = sm_load.to(device)
        enz_load = enz_load.to(device)
        y = y.to(device)

        pred = model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# 9. 主程序
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)


    # ========================================================
    # 特征
    # ========================================================

    with open(ESM_PKL, "rb") as f:
        esm_dict = pickle.load(f)

    with open(MOL_PKL, "rb") as f:
        mol_dict = pickle.load(f)

    print(
        f"ESM={len(esm_dict)}, "
        f"MolFormer={len(mol_dict)}"
    )


    # ========================================================
    # split
    # ========================================================

    def read_split(name):

        with open(os.path.join(SPLIT_DIR, name)) as f:
            return [x.strip() for x in f if x.strip()]

    train_idx = read_split(TRAIN_TXT)
    val_idx = read_split(VAL_TXT)
    test_idx = read_split(TEST_TXT)

    # ========================================================
    # CSV
    # ========================================================

    df = pd.read_csv(CSV_PATH)

    df["idx"] = df["idx"].astype(str)

    df["Activity"] = pd.to_numeric(df["Activity"],errors="coerce")

    df = df.dropna(subset=["Activity"]).reset_index(drop=True)

    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)

    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)

    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)


    print(
        f"train={len(df_train)}, "
        f"val={len(df_val)}, "
        f"test={len(df_test)}"
    )


    # ========================================================
    # 浓度处理
    # 和原模型保持一致
    # ========================================================

    SM_COL = "SM loading (g/L)"
    ENZ_COL = "Enzyme loading (g/L)"

    for d in [df_train,df_val,df_test]:

        d[SM_COL] = pd.to_numeric(d[SM_COL],errors="coerce").fillna(0).clip(lower=0)

        d[ENZ_COL] = pd.to_numeric(d[ENZ_COL],errors="coerce").fillna(0).clip(lower=0)

        d["SM_log"] = np.log1p(d[SM_COL])

        d["Enz_log"] = np.log1p(d[ENZ_COL])


    sm_mean = df_train["SM_log"].mean()
    sm_std = df_train["SM_log"].std() + 1e-12

    enz_mean = df_train["Enz_log"].mean()
    enz_std = df_train["Enz_log"].std() + 1e-12


    for d in [df_train,df_val,df_test]:

        d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
        d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std


 # ========================================================
    # Activity标签 标准化
    # 只使用 train
    #scaler会保存{
#     "activity_mean":0.037,
#     "activity_std":0.121,
#     "activity_threshold":0.03
# }推理时要用，最后要反标准化回来，pred_activity = pred_norm * 0.121 + 0.037≈0.182   > 0.03 => class=1
    # ========================================================

    activity_mean = df_train["Activity"].mean()
    activity_std = (df_train["Activity"].std()+ 1e-12)

    print("\nActivity mean:",activity_mean)

    print("Activity std:",activity_std)
    print(df["Activity"].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

    # 保存 scaler
    pd.Series({
        "activity_mean": activity_mean,
        "activity_std": activity_std,
        "activity_threshold": ACTIVITY_THRESHOLD
    }).to_json(
        os.path.join(
            MODEL_DIR,
            "activity_scaler.json"
        )
    )


    # ========================================================
    # Dataset
    # ========================================================

    train_set = EnzymeSubstrateDataset(df_train,esm_dict,mol_dict,activity_mean,activity_std)

    val_set = EnzymeSubstrateDataset(df_val,esm_dict,mol_dict,activity_mean,activity_std)

    test_set = EnzymeSubstrateDataset(df_test,esm_dict,mol_dict,activity_mean,activity_std)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

 # ========================================================
    # 模型
    # ========================================================

    model = EnzymeSubstrateRegressor().to(device)

    model = load_pretrained(
        model,
        PRETRAINED_MODEL,
        device)

    criterion = nn.SmoothL1Loss()


    # ========================================================
    # Stage 1
    # 只训练 fc4
    # ========================================================

    print("\n========== Stage 1 ==========")

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc4.parameters():
        p.requires_grad = True   #只解冻fc4

    optimizer = torch.optim.AdamW(model.fc4.parameters(),lr=1e-3,weight_decay=1e-5)  #只优化初始化的fc4权重

    best_f1 = -np.inf

    for epoch in range(50):

        loss = train_epoch(model,train_loader,optimizer,criterion,device)

        true_val, pred_val = predict(model,val_loader,device,activity_mean,activity_std)

        m = evaluate_binary(true_val,pred_val)

        print(
            f"Stage1 {epoch+1:03d}/50 | "
            f"Loss={loss:.5f} | "
            f"R2={m['R2']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"AUC={m['AUC']:.4f}")

        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            torch.save(model.state_dict(),os.path.join(MODEL_DIR,"stage1_best.pth"))


    # 加载 Stage1 最优
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"stage1_best.pth"),map_location=device))

 # ========================================================
    # Stage 2
    # 全模型低学习率微调
    # ========================================================

    print("\n========== Stage 2 ==========")

    for p in model.parameters():
        p.requires_grad = True  #解冻全部层

    optimizer = torch.optim.AdamW(    #低学习率微调,把转化率表征逐渐调整成活性表征
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-5
    )

    best_f1 = -np.inf

    best_path = os.path.join(MODEL_DIR,"best_model.pth")


    for epoch in range(500):
        loss = train_epoch(model,train_loader,optimizer,criterion,device)
        true_val, pred_val = predict(model,val_loader,device,activity_mean,activity_std)

        m = evaluate_binary(true_val,pred_val)

        print(
            f"Stage2 {epoch+1:03d}/500 | "
            f"Loss={loss:.5f} | "
            f"RMSE={m['RMSE']:.5f} | "
            f"MAE={m['MAE']:.5f} | "
            f"R2={m['R2']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"AUC={m['AUC']:.4f}"
        )

        # 最终目的是 0.03 二分类，
        # 所以这里按照 F1 保存最佳模型
        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            torch.save(model.state_dict(),best_path)

            print(f"  >>> Best model: F1={best_f1:.4f}")


    # ========================================================
    # Test
    # ========================================================

    model.load_state_dict(torch.load(best_path,map_location=device))


    true_test, pred_test = predict(model,test_loader,device,activity_mean,activity_std)

    test_m = evaluate_binary(true_test,pred_test)

    print("\n")
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    for k, v in test_m.items():
        print(f"{k}: {v}")


    # ========================================================
    # 保存最终预测
    # ========================================================

    result = df_test.copy()

    result["True_Activity"] = true_test
    result["Pred_Activity"] = pred_test

    result["True_Binary"] = (result["True_Activity"]>= ACTIVITY_THRESHOLD).astype(int)

    result["Pred_Binary"] = (result["Pred_Activity"]>= ACTIVITY_THRESHOLD).astype(int)

    result.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"),index=False,encoding="utf-8-sig")


    print("\n预测结果已保存：",os.path.join(MODEL_DIR,"test_predictions.csv"))


if __name__ == "__main__":
    main()

