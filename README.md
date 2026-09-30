#                  ┌───────────────┐
#                  │   ESM2        │
#                  └───────┬───────┘
#                          │
#                          ↓
#                  ┌───────────────┐
#                  │ Transformer   │
#                  └───────┬───────┘
#                          │
#                          │
# MolFormer ───────────────┤
#                          │
# Enzyme concentration ───┤
#                          │
# Substrate concentration ┤
#                          ↓
#                 Multimodal Fusion
#                          │
#                          ↓
#                   FC1 → FC2 → FC3
#                          │
#                          ↓
#                   Binary Classifier
#                          │
#                          ↓
#                     sigmoid
#                          │
#                          ↓
#                  P(Active)
#                   /          \
#               < 0.5          ≥ 0.5
#                 ↓              ↓
#                 0              1
#               无活性          有活性

import os
import random
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 搜索最佳阈值
# ============================================================

def find_best_threshold(y_true, y_prob):
    best_th, best_f1 = 0.5, -1

    for th in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1
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
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/splits_Activity_high"

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"

# 输出
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_finetune_high_cls_2"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Activity 二分类阈值
# ============================================================

ACTIVITY_THRESHOLD = 1

# ============================================================
# Stage 1
# ============================================================

STAGE1_EPOCHS = 50
STAGE1_LR = 1e-3

# ============================================================
# Stage 2
# ============================================================

STAGE2_EPOCHS = 500
STAGE2_LR = 3e-5  #5e-5

WEIGHT_DECAY = 1e-5


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed=123):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Dataset
# ============================================================
class EnzymeSubstrateDataset(Dataset):

    def __init__(self, df, esm_embeds, mol_embeds):

        self.df = df.reset_index(drop=True).copy()

        self.idx = self.df["Name"].astype(str).tolist()
        self.smiles = self.df["SM"].tolist()

        activity = self.df["Activity"].astype(float).values
        self.labels = (activity >= ACTIVITY_THRESHOLD).astype(np.float32)

        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        # 归一化后浓度列（log+zscore，在 main() 中计算）
        if "SM_norm" not in self.df.columns or "Enz_norm" not in self.df.columns:
            raise ValueError("DataFrame 必须包含 'SM_norm' 和 'Enz_norm'")

        self.sm_norm = self.df["SM_norm"].astype(float).values
        self.enzyme_norm = self.df["Enz_norm"].astype(float).values

        missing_esm = [x for x in self.idx if x not in esm_embeds]
        if missing_esm:
            raise KeyError(
                f"ESM.pkl 中缺少 {len(missing_esm)} 个 Name，例如：{missing_esm[:5]}"
            )

        missing_mol = [x for x in self.smiles if x not in mol_embeds]
        if missing_mol:
            raise KeyError(
                f"SM.pkl 中缺少 {len(missing_mol)} 个 SM，例如：{missing_mol[:5]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        esm_data = self.esm_embeds[self.idx[i]]

        esm_emb = esm_data["emb"].float()
        esm_mask = esm_data["mask"].long()

        mol_feat = torch.as_tensor(
            self.mol_embeds[self.smiles[i]],
            dtype=torch.float32
        )

        sm_load = torch.tensor([self.sm_norm[i]], dtype=torch.float32)
        enz_load = torch.tensor([self.enzyme_norm[i]], dtype=torch.float32)

        label = torch.tensor(self.labels[i],dtype=torch.float32)

        return (esm_emb,esm_mask,mol_feat,sm_load,enz_load,label)


# ============================================================
# 4. 模型
# ============================================================

class EnzymeSubstrateBinaryClassifier(nn.Module):

    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()

        # ESM & MolFormer
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        # Transformer
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

        # Concentration
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

        input_dim = 256 + 256 + 128 + 128

        # Fusion MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(0.2)

        # Binary head
        self.classifier = nn.Linear(hidden_dim // 2, 1)


    def forward(self,esm_feat,esm_mask,mol_feat,sm_load,enz_load):

        # ESM encoder
        esm_feat = self.esm_proj(esm_feat)

        src_mask = (esm_mask == 0).bool()

        esm_feat = self.esm_encoder(esm_feat,src_key_padding_mask=src_mask)

        # Masked mean pooling
        mask = esm_mask.unsqueeze(-1).float()

        esm_pooled = (esm_feat * mask).sum(1) / (mask.sum(1).clamp(min=1e-8))

        # Other modalities
        mol_proj = self.mol_proj(mol_feat)
        sm_proj = self.sm_proj(sm_load)
        enz_proj = self.enz_proj(enz_load)

        # Fusion
        x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj],dim=-1)

        # FC blocks
        x = F.relu(self.bn1(self.fc1(x)))

        residual = x

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual

        x = F.relu(self.bn3(self.fc3(x)))

        logits = self.classifier(x).squeeze(-1)

        return logits


# ============================================================
# 5. 加载原始回归模型
# ============================================================

def load_pretrained(model, path, device):

    print("\n========== Loading pretrained model ==========")

    ckpt = torch.load(path, map_location=device)  #读取预训练权重

    if isinstance(ckpt, dict) and "state_dict" in ckpt: #兼容不同保存格式
        ckpt = ckpt["state_dict"]

    # 去掉 DataParallel 的 module. 例如将model = nn.DataParallel(model)保存得到module.fc1.weight 变成fc1.weight
    ckpt = {
        k.replace("module.", ""): v
        for k, v in ckpt.items()
    }

    # 不加载旧回归头
    ckpt.pop("fc4.weight", None)
    ckpt.pop("fc4.bias", None)

    # 加载预训练其余·权重
    missing, unexpected = model.load_state_dict(ckpt,strict=False)

    print("Missing keys:")
    for k in missing:
        print(" ", k)

    print("Unexpected keys:")
    for k in unexpected:
        print(" ", k)

    # 初始化新的分类头
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    print("\n✅ 原模型 backbone 加载完成")
    print("✅ 原 fc4 回归头未加载")
    print("✅ 新 classifier 二分类头已初始化")

    return model

# ============================================================
# 6. 训练一个 epoch
# ============================================================
def train_epoch(model,loader,optimizer,criterion,device,train_mode=True):

    model.train() if train_mode else model.eval()
    total_loss = 0.0
    total_samples = 0

    for (esm_feat,esm_mask,mol_feat,sm_load,enz_load,label) in loader:

        esm_feat = esm_feat.to(device, non_blocking=True)
        esm_mask = esm_mask.to(device, non_blocking=True)
        mol_feat = mol_feat.to(device, non_blocking=True)
        sm_load = sm_load.to(device, non_blocking=True)
        enz_load = enz_load.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)

        loss = criterion(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)

        optimizer.step()

        batch_size = label.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# 7. Prediction
# ============================================================

def predict(model,loader,device):

    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for (esm_feat,esm_mask,mol_feat,sm_load,enz_load,label) in loader:
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)
            # ------------------------------------------------
            # logits
            # ------------------------------------------------
            logits = model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)
            # ------------------------------------------------
            # sigmoid
            # ------------------------------------------------
            prob = torch.sigmoid(logits)
            probabilities.extend(prob.cpu().numpy())
            labels.extend(label.numpy())

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels).astype(int)
    predictions = (probabilities >= 0.5).astype(int)

    #return (labels,probabilities,predictions)  #固定0.5阈值
    return (labels,probabilities)


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate(true,probabilities,predictions):

    accuracy = accuracy_score(true,predictions)

    precision = precision_score(true,predictions,zero_division=0)

    recall = recall_score(true,predictions,zero_division=0)

    f1 = f1_score(true,predictions,zero_division=0)

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------
    if len(np.unique(true)) == 2:
        auc = roc_auc_score(true,probabilities)
    else:
        auc = np.nan

    # --------------------------------------------------------
    # confusion matrix
    # --------------------------------------------------------
    cm = confusion_matrix(true,predictions,labels=[0, 1])

    return {
        "Accuracy": accuracy,
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
# 9. 主程序
# ============================================================

def main():

  
    set_seed(SEED)

    device = torch.device("cuda"if torch.cuda.is_available()else "cpu")
    print("Using device:",device)

    print("\nLoading ESM features...")
    with open(ESM_PKL,"rb") as f:
        esm_dict = pickle.load(f)


    print("Loading MolFormer features...")
    with open(MOL_PKL,"rb") as f:
        mol_dict = pickle.load(f)

    print(f"ESM={len(esm_dict)}")

    print(f"MolFormer={len(mol_dict)}")

    # ========================================================
    # split
    # ========================================================
    def read_split(filename):

        path = os.path.join(SPLIT_DIR,filename)

        with open(path) as f:
            return [x.strip() for x in f if x.strip()]

    train_idx = set(read_split(TRAIN_TXT))

    val_idx = set(read_split(VAL_TXT))

    test_idx = set(read_split(TEST_TXT))

    print("\nSplit:")

    print("Train:",len(train_idx))

    print("Val:",len(val_idx))

    print("Test:",len(test_idx))

    # ========================================================
    # 检查列
    # ========================================================
    print("\nLoading CSV...")
    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "idx",
        "Name",
        "SM",
        "Activity",
        "SM loading (g/L)",
        "Enzyme loading (g/L)"
    ]

    missing_columns = [
        x for x in required_columns
        if x not in df.columns
    ]

    if missing_columns:
        raise ValueError("CSV 缺少以下列："+ str(missing_columns))

    # ========================================================
    # ID
    # ========================================================

    df["idx"] = (df["idx"].astype(str))

    df["Name"] = (df["Name"].astype(str))

    df["Activity"] = pd.to_numeric(df["Activity"],errors="coerce")

    raw_size = len(df)

    missing_activity = (df["Activity"].isna().sum())

    print("\nRaw data:",raw_size)

    print("Activity NaN:",missing_activity)

    # 删除 Activity 缺失
    df = df.dropna(subset=["Activity"]).reset_index(drop=True)
    print("Valid data:",len(df))

    # ========================================================
    # 根据Split分数据
    # ========================================================

    df_train = df[df["idx"].isin(train_idx)].copy().reset_index(drop=True)
    df_val   = df[df["idx"].isin(val_idx)].copy().reset_index(drop=True)
    df_test  = df[df["idx"].isin(test_idx)].copy().reset_index(drop=True)

    print(f"\nDataset size:\n"
        f"Train: {len(df_train)}\n"
        f"Val:   {len(df_val)}\n"
        f"Test:  {len(df_test)}")

    # ========================================================
    # Binary distribution
    # ========================================================

    print("\n========== Binary distribution ==========")

    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:

        y = (d["Activity"] >= ACTIVITY_THRESHOLD).astype(int)

        n0 = (y == 0).sum()
        n1 = (y == 1).sum()

        print(
            f"{name}: "
            f"0={n0} ({n0/len(y):.2%}) | "
            f"1={n1} ({n1/len(y):.2%})")

    # ========================================================
    # Concentration preprocessing
    # ========================================================

    SM_COL = "SM loading (g/L)"
    ENZ_COL = "Enzyme loading (g/L)"

    for d in [df_train, df_val, df_test]:

        d[SM_COL] = (pd.to_numeric(d[SM_COL], errors="coerce").fillna(0).clip(lower=0))
        d[ENZ_COL] = (pd.to_numeric(d[ENZ_COL], errors="coerce").fillna(0).clip(lower=0))
        d["SM_log"] = np.log1p(d[SM_COL])
        d["Enz_log"] = np.log1p(d[ENZ_COL])

    # ========================================================
    # Scaler (train only)
    # ========================================================

    sm_mean = df_train["SM_log"].mean()
    sm_std  = df_train["SM_log"].std() + 1e-12

    enz_mean = df_train["Enz_log"].mean()
    enz_std  = df_train["Enz_log"].std() + 1e-12

    for d in [df_train, df_val, df_test]:

        d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
        d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std

    # ========================================================
    # Save scaler
    # ========================================================

    scaler = {
        "sm_mean": float(sm_mean),
        "sm_std": float(sm_std),
        "enz_mean": float(enz_mean),
        "enz_std": float(enz_std),
        "activity_threshold": float(ACTIVITY_THRESHOLD)
    }

    pd.Series(scaler).to_json(os.path.join(MODEL_DIR, "scaler.json"))

    # ========================================================
    # Dataset &  DataLoader
    # ========================================================

    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict)
    val_set   = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict)
    test_set  = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
  
    # ========================================================
    # Model
    # ========================================================
    model = EnzymeSubstrateBinaryClassifier().to(device)
    model = load_pretrained(model,PRETRAINED_MODEL,device)

    # ========================================================
    # 计算 pos_weight
    # ========================================================

    train_y = (df_train["Activity"]>= ACTIVITY_THRESHOLD).astype(int)

    positive = (train_y == 1).sum()

    negative = (train_y == 0).sum()

    if positive == 0:

        raise ValueError("训练集没有正样本。")

    pos_weight_value = (negative / positive) * 1.5

    print("\n========== Class weight ==========")

    print("Negative:",negative)
    print("Positive:",positive)
    print("pos_weight:",pos_weight_value)

    criterion = (nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value,dtype=torch.float32,device=device)))

 
    # ========================================================
    # Stage 1 (把“固定0.5阈值 + 固定学习率 + 固定500轮训练”改成了“自动找最佳阈值 + 自动降学习率 + 自动早停)
    # ========================================================

    print("\n========== Stage 1 只训练分类头 ==========")

    for p in model.parameters():  #冻结整个模型
        p.requires_grad = False

    for p in model.classifier.parameters():   #只打开分类头，解冻
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.classifier.parameters(),lr=STAGE1_LR,weight_decay=WEIGHT_DECAY)

    best_f1 = -1
    stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")

    for epoch in range(STAGE1_EPOCHS):
        loss = train_epoch(model,train_loader,optimizer,criterion,device,train_mode=False)

        true_val, prob_val = predict(model,val_loader,device)

        pred_val = (prob_val >= 0.5).astype(int)

        m = evaluate(true_val,prob_val,pred_val)

        print(
            f"Stage1 {epoch+1:03d}/{STAGE1_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"F1={m['F1']:.4f} | "
            f"P={m['Precision']:.4f} | "
            f"R={m['Recall']:.4f} | "
            f"AUC={m['AUC']:.4f}")

        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            torch.save(model.state_dict(), stage1_path)

    print(f"\nStage1 Best F1={best_f1:.4f}")

    model.load_state_dict(
        torch.load(stage1_path, map_location=device)
    )

    # ========================================================
    # Stage 2 
    # ========================================================

    print("\n========== Stage 2 全模型微调 ==========")

    for p in model.parameters():   #全部解冻
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(),lr=STAGE2_LR,weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=15
    )

    best_f1 = -1
    best_threshold = 0.5

    patience = 30
    early_counter = 0

    best_path = os.path.join(MODEL_DIR, "best_model.pth")

    for epoch in range(STAGE2_EPOCHS):
        loss = train_epoch( model,train_loader,optimizer,criterion,device,train_mode=True)
        true_val, prob_val = predict(model,val_loader,device)

        cur_threshold, _ = find_best_threshold(true_val,prob_val)

        pred_val = (prob_val >= cur_threshold).astype(int)

        m = evaluate(true_val,prob_val,pred_val)
        scheduler.step(m["F1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Stage2 {epoch+1:03d}/{STAGE2_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"F1={m['F1']:.4f} | "
            f"P={m['Precision']:.4f} | "
            f"R={m['Recall']:.4f} | "
            f"AUC={m['AUC']:.4f} | "
            f"TH={cur_threshold:.2f} | "
            f"LR={current_lr:.2e}")

        if m["F1"] > best_f1:

            best_f1 = m["F1"]
            best_threshold = cur_threshold
            early_counter = 0

            torch.save({"model": model.state_dict(),"threshold": best_threshold},best_path)

            print(f">>> Best F1={best_f1:.4f} " f"TH={best_threshold:.2f}")

        else:
            early_counter += 1

        if early_counter >= patience:
            print(f"\nEarly Stopping at Epoch {epoch+1}")
            break

    # ========================================================
    # Final Test
    # ========================================================

    print("\n========== FINAL TEST ==========")

    ckpt = torch.load(
        best_path,
        map_location=device
    )

    model.load_state_dict(ckpt["model"])

    best_threshold = ckpt["threshold"]

    print(f"Best Threshold = {best_threshold:.2f}")

    true_test, prob_test = predict(model,test_loader,device)

    pred_test = (prob_test >= best_threshold).astype(int)

    test_m = evaluate(true_test,prob_test,pred_test)

    print("\n========== TEST RESULT ==========")

    for k, v in test_m.items():
        print(f"{k}: {v}")

    # ========================================================
    # Save Prediction
    # ========================================================

    result = df_test.copy()

    result["True_Activity"] = result["Activity"]

    result["True_Binary"] = (result["Activity"] >= ACTIVITY_THRESHOLD).astype(int)

    result["Pred_Active_Probability"] = prob_test
    result["Pred_Binary"] = pred_test

    output_csv = os.path.join(MODEL_DIR,"test_predictions.csv")

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nPrediction saved: {output_csv}")


    
    # # ========================================================
    # # Stage 1(固定0.5阈值 + 固定学习率 + 固定500轮训练)
    # # 只训练新的 classifier
    # # ========================================================

    # print("\n========== Stage 1 ==========")

    # for p in model.parameters():
    #     p.requires_grad = False

    # for p in model.classifier.parameters():
    #     p.requires_grad = True

    # optimizer = torch.optim.AdamW(model.classifier.parameters(),lr=STAGE1_LR,weight_decay=WEIGHT_DECAY)

    # best_f1 = -np.inf
    # stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")

    # for epoch in range(STAGE1_EPOCHS):
    #     loss = train_epoch(model,train_loader,optimizer,criterion,device,train_mode=False)
    #     true_val, prob_val, pred_val = predict(model,val_loader,device)

    #     m = evaluate(true_val, prob_val, pred_val)

    #     print(
    #         f"Stage1 {epoch+1:03d}/{STAGE1_EPOCHS} | "
    #         f"Loss={loss:.5f} | "
    #         f"F1={m['F1']:.4f} | "
    #         f"P={m['Precision']:.4f} | "
    #         f"R={m['Recall']:.4f} | "
    #         f"AUC={m['AUC']:.4f}")

    #     if m["F1"] > best_f1:
    #         best_f1 = m["F1"]
    #         torch.save(model.state_dict(), stage1_path)

    # print(f"\nStage1 Best F1={best_f1:.4f}")

    # model.load_state_dict(torch.load(stage1_path, map_location=device))

    # # ========================================================
    # # Stage 2
    # # ========================================================

    # print("\n========== Stage 2 ==========")

    # for p in model.parameters():
    #     p.requires_grad = True

    # optimizer = torch.optim.AdamW(model.parameters(),lr=STAGE2_LR,weight_decay=WEIGHT_DECAY)

    # best_f1 = -np.inf
    # best_path = os.path.join(MODEL_DIR, "best_model.pth")

    # for epoch in range(STAGE2_EPOCHS):
    #     loss = train_epoch(model,train_loader,optimizer,criterion,device,train_mode=True)
    #     true_val, prob_val, pred_val = predict(model,val_loader,device)
    #     m = evaluate(true_val, prob_val, pred_val)

    #     print(
    #         f"Stage2 {epoch+1:03d}/{STAGE2_EPOCHS} | "
    #         f"Loss={loss:.5f} | "
    #         f"F1={m['F1']:.4f} | "
    #         f"P={m['Precision']:.4f} | "
    #         f"R={m['Recall']:.4f} | "
    #         f"AUC={m['AUC']:.4f}")

    #     if m["F1"] > best_f1:
    #         best_f1 = m["F1"]
    #         torch.save(model.state_dict(), best_path)
    #         print(f">>> Best F1={best_f1:.4f}")

    # # ========================================================
    # # Final Test
    # # ========================================================

    # print("\n========== FINAL TEST ==========")

    # model.load_state_dict(torch.load(best_path, map_location=device))

    # true_test, prob_test, pred_test = predict(model,test_loader,device)
    # test_m = evaluate(true_test,prob_test,pred_test)

    # for k, v in test_m.items():
    #     print(f"{k}: {v}")

    # # ========================================================
    # # Save prediction
    # # ========================================================

    # result = df_test.copy()

    # result["True_Activity"] = result["Activity"]
    # result["True_Binary"] = (result["Activity"] >= ACTIVITY_THRESHOLD).astype(int)

    # result["Pred_Active_Probability"] = prob_test
    # result["Pred_Binary"] = pred_test

    # output_csv = os.path.join(MODEL_DIR,"test_predictions.csv")
    # result.to_csv(output_csv,index=False,encoding="utf-8-sig")
    # print(f"\n预测结果已保存: {output_csv}")


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    main()
