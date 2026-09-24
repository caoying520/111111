import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel


# ============================================================
# 1. 路径
# ============================================================

MODEL_PATH = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/best_model.pth"
)

SCALER_PATH = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/scaler.json"
)

INPUT_CSV = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/predict_data.csv"
)

ESM_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/esm_feats_emb_mask_test_6_0.pkl"
)

MOL_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/SM.pkl"
)

OUTPUT_CSV = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/predictions.csv"
)


# ============================================================
# 2. 参数
# ============================================================

BATCH_SIZE = 256

# 注意：
# 这里是“模型输出概率”的二分类阈值
# 不是 Activity=0.03
PRED_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
print("Prediction probability threshold:", PRED_THRESHOLD)


# ============================================================
# 3. Dataset
# ============================================================

class ActivityPredictionDataset(Dataset):

    def __init__(
        self,
        df,
        esm_embeds,
        mol_embeds,
        sm_mean,
        sm_std,
        enz_mean,
        enz_std
    ):

        self.df = df.reset_index(drop=True)
        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        self.sm_mean = sm_mean
        self.sm_std = sm_std
        self.enz_mean = enz_mean
        self.enz_std = enz_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        name = str(row["Name"])
        sm = str(row["SM"])

        # ----------------------------------------------------
        # ESM
        # ----------------------------------------------------
        esm_data = self.esm_embeds[name]

        esm_emb = esm_data["emb"]
        esm_mask = esm_data["mask"]

        if not torch.is_tensor(esm_emb):
            esm_emb = torch.tensor(esm_emb, dtype=torch.float32)
        else:
            esm_emb = esm_emb.float()

        if not torch.is_tensor(esm_mask):
            esm_mask = torch.tensor(esm_mask, dtype=torch.bool)
        else:
            esm_mask = esm_mask.bool()

        # ----------------------------------------------------
        # MolFormer
        # ----------------------------------------------------
        mol_emb = self.mol_embeds[sm]

        if not torch.is_tensor(mol_emb):
            mol_emb = torch.tensor(mol_emb, dtype=torch.float32)
        else:
            mol_emb = mol_emb.float()

        # ----------------------------------------------------
        # substrate concentration
        # ----------------------------------------------------
        sm_loading = pd.to_numeric(
            row["SM loading (g/L)"],
            errors="coerce"
        )

        if pd.isna(sm_loading):
            sm_loading = 0.0

        sm_loading = max(float(sm_loading), 0.0)

        # 与训练时保持一致
        sm_loading = np.log1p(sm_loading)

        sm_norm = (
            sm_loading - self.sm_mean
        ) / self.sm_std

        # ----------------------------------------------------
        # enzyme concentration
        # ----------------------------------------------------
        enz_loading = pd.to_numeric(
            row["Enzyme loading (g/L)"],
            errors="coerce"
        )

        if pd.isna(enz_loading):
            enz_loading = 0.0

        enz_loading = max(float(enz_loading), 0.0)

        # 与训练时保持一致
        enz_loading = np.log1p(enz_loading)

        enz_norm = (
            enz_loading - self.enz_mean
        ) / self.enz_std

        return (
            esm_emb,
            esm_mask,
            mol_emb,
            torch.tensor(sm_norm, dtype=torch.float32),
            torch.tensor(enz_norm, dtype=torch.float32),
            name,
            sm
        )


# ============================================================
# 4. 模型
# ============================================================

class EnzymeSubstrateBinaryClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # ESM
        # ----------------------------------------------------
        self.esm_fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ----------------------------------------------------
        # MolFormer
        # ----------------------------------------------------
        self.mol_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # substrate concentration
        # ----------------------------------------------------
        self.sm_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # ----------------------------------------------------
        # enzyme concentration
        # ----------------------------------------------------
        self.enz_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # ----------------------------------------------------
        # fusion
        # 256 + 256 + 128 + 128 = 768
        # ----------------------------------------------------
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.2)

        # 二分类输出
        self.classifier = nn.Linear(128, 1)

    def forward(
        self,
        esm_emb,
        esm_mask,
        mol_emb,
        sm_norm,
        enz_norm
    ):

        # ----------------------------------------------------
        # ESM
        # ----------------------------------------------------
        x = self.esm_fc(esm_emb)

        # Transformer padding mask
        # True = padding
        src_key_padding_mask = ~esm_mask.bool()

        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # mask mean pooling
        mask = esm_mask.unsqueeze(-1).float()

        x = x * mask

        x = x.sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1.0)

        esm_feature = x

        # ----------------------------------------------------
        # MolFormer
        # ----------------------------------------------------
        mol_feature = self.mol_fc(mol_emb)

        # ----------------------------------------------------
        # concentrations
        # ----------------------------------------------------
        sm_feature = self.sm_fc(
            sm_norm.unsqueeze(1)
        )

        enz_feature = self.enz_fc(
            enz_norm.unsqueeze(1)
        )

        # ----------------------------------------------------
        # fusion
        # ----------------------------------------------------
        fusion = torch.cat(
            [
                esm_feature,
                mol_feature,
                sm_feature,
                enz_feature
            ],
            dim=1
        )

        x = self.fc1(fusion)

        # residual
        residual = x

        x = self.fc2(x)

        x = x + residual

        x = self.fc3(x)

        x = self.dropout(x)

        # logits
        logits = self.classifier(x)

        return logits.squeeze(1)


# ============================================================
# 5. 读取 scaler
# ============================================================

print("\nLoading scaler...")

with open(SCALER_PATH, "r") as f:
    scaler = json.load(f)

sm_mean = scaler["sm_mean"]
sm_std = scaler["sm_std"]

enz_mean = scaler["enz_mean"]
enz_std = scaler["enz_std"]

print("SM mean:", sm_mean)
print("SM std :", sm_std)
print("Enz mean:", enz_mean)
print("Enz std :", enz_std)


# ============================================================
# 6. 读取 ESM / MolFormer 特征
# ============================================================

print("\nLoading ESM embeddings...")

with open(ESM_PKL, "rb") as f:
    esm_embeds = pickle.load(f)

print("ESM embeddings:", len(esm_embeds))


print("\nLoading MolFormer embeddings...")

with open(MOL_PKL, "rb") as f:
    mol_embeds = pickle.load(f)

print("MolFormer embeddings:", len(mol_embeds))


# ============================================================
# 7. 读取预测数据
# ============================================================

print("\nLoading input CSV...")

df = pd.read_csv(INPUT_CSV)

required_columns = [
    "Name",
    "SM",
    "SM loading (g/L)",
    "Enzyme loading (g/L)"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Input CSV 缺少列: {col}"
        )

print("Prediction samples:", len(df))


# ============================================================
# 8. 检查特征是否存在
# ============================================================

missing_esm = [
    name for name in df["Name"].astype(str)
    if name not in esm_embeds
]

missing_mol = [
    sm for sm in df["SM"].astype(str)
    if sm not in mol_embeds
]

if len(missing_esm) > 0:
    print(
        f"\nWARNING: {len(missing_esm)} 个 Name 没有 ESM 特征"
    )
    print("前10个:", missing_esm[:10])

if len(missing_mol) > 0:
    print(
        f"\nWARNING: {len(missing_mol)} 个 SM 没有 MolFormer 特征"
    )
    print("前10个:", missing_mol[:10])


# 只保留特征完整的数据
valid_mask = (
    df["Name"].astype(str).isin(esm_embeds.keys())
    &
    df["SM"].astype(str).isin(mol_embeds.keys())
)

df_valid = df[valid_mask].copy()

df_valid = df_valid.reset_index(drop=True)

print(
    "\nValid samples:",
    len(df_valid),
    "/",
    len(df)
)


# ============================================================
# 9. Dataset / DataLoader
# ============================================================

dataset = ActivityPredictionDataset(
    df_valid,
    esm_embeds,
    mol_embeds,
    sm_mean,
    sm_std,
    enz_mean,
    enz_std
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 10. 创建模型
# ============================================================

print("\nCreating model...")

model = EnzymeSubstrateBinaryClassifier()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# 当前 best_model.pth 是纯 state_dict
if isinstance(checkpoint, dict) and "model" in checkpoint:
    print("检测到 checkpoint 格式：{'model': ...}")
    state_dict = checkpoint["model"]
else:
    print("检测到 checkpoint 格式：纯 state_dict")
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# 11. 预测
# ============================================================

all_probabilities = []

print("\nStart prediction...")

with torch.no_grad():

    for batch in loader:

        (
            esm_emb,
            esm_mask,
            mol_emb,
            sm_norm,
            enz_norm,
            names,
            sms
        ) = batch

        esm_emb = esm_emb.to(DEVICE)
        esm_mask = esm_mask.to(DEVICE)

        mol_emb = mol_emb.to(DEVICE)

        sm_norm = sm_norm.to(DEVICE)
        enz_norm = enz_norm.to(DEVICE)

        logits = model(
            esm_emb,
            esm_mask,
            mol_emb,
            sm_norm,
            enz_norm
        )

        probabilities = torch.sigmoid(logits)

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )


all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# 12. 二分类
# ============================================================

pred_binary = (
    all_probabilities >= PRED_THRESHOLD
).astype(int)


# ============================================================
# 13. 保存结果
# ============================================================

result = df_valid.copy()

result["Pred_Active_Probability"] = all_probabilities

result["Pred_Binary"] = pred_binary

# 方便查看中文结果
result["Pred_Result"] = np.where(
    pred_binary == 1,
    "有活性",
    "无活性"
)


# 如果原始数据里面有 Activity，
# 则同时计算真实标签
if "Activity" in result.columns:

    result["Activity"] = pd.to_numeric(
        result["Activity"],
        errors="coerce"
    )

    result["True_Binary"] = (
        result["Activity"] >= 0.03
    ).astype(int)


result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 输出统计
# ============================================================

print("\nPrediction finished.")

print("Output:", OUTPUT_CSV)

print("\nPrediction statistics:")
print(
    "无活性(0):",
    int((pred_binary == 0).sum())
)

print(
    "有活性(1):",
    int((pred_binary == 1).sum())
)

print(
    "有活性比例:",
    f"{pred_binary.mean():.4f}"
)

print(
    "Probability min:",
    all_probabilities.min()
)

print(
    "Probability max:",
    all_probabilities.max()
)

print(
    "Probability mean:",
    all_probabilities.mean()
)

print("\n前10条预测结果：")

show_cols = [
    "Name",
    "SM",
    "Pred_Active_Probability",
    "Pred_Binary",
    "Pred_Result"
]

print(
    result[show_cols].head(10).to_string(index=False)
)
