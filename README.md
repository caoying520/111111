import os
import json
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 参数
# ============================================================

BATCH_SIZE = 256

# ------------------------------------------------------------
# 训练好的二分类模型
# ------------------------------------------------------------
MODEL_PATH = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/best_model.pth"
)

# ------------------------------------------------------------
# 训练时保存的 scaler
# ------------------------------------------------------------
SCALER_PATH = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/scaler.json"
)

# ------------------------------------------------------------
# 待预测 CSV
#
# 至少需要：
# Name
# SM
# SM loading (g/L)
# Enzyme loading (g/L)
#
# 如果有 Activity 也可以，没有也可以
# ------------------------------------------------------------
INPUT_CSV = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/predict_data.csv"
)

# ------------------------------------------------------------
# ESM 特征
# ------------------------------------------------------------
ESM_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/esm_feats_emb_mask_test_6_0.pkl"
)

# ------------------------------------------------------------
# MolFormer 特征
# ------------------------------------------------------------
MOL_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/SM.pkl"
)

# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------
OUTPUT_CSV = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_finetune_cls_2/predictions.csv"
)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Dataset
# ============================================================

class PredictionDataset(Dataset):

    def __init__(
        self,
        df,
        esm_embeds,
        mol_embeds
    ):

        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        self.names = (
            self.df["Name"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # SMILES
        # ----------------------------------------------------

        self.smiles = (
            self.df["SM"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # 浓度
        # ----------------------------------------------------

        self.sm_norm = (
            self.df["SM_norm"]
            .astype(float)
            .values
        )

        self.enzyme_norm = (
            self.df["Enz_norm"]
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # 检查 ESM
        # ----------------------------------------------------

        missing_esm = [
            x for x in self.names
            if x not in self.esm_embeds
        ]

        if len(missing_esm) > 0:

            raise KeyError(
                f"ESM 特征中缺少 {len(missing_esm)} 个 Name。\n"
                f"例如：{missing_esm[:10]}"
            )

        # ----------------------------------------------------
        # 检查 MolFormer
        # ----------------------------------------------------

        missing_mol = [
            x for x in self.smiles
            if x not in self.mol_embeds
        ]

        if len(missing_mol) > 0:

            raise KeyError(
                f"MolFormer 特征中缺少 {len(missing_mol)} 个 SM。\n"
                f"例如：{missing_mol[:10]}"
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, i):

        # ====================================================
        # ESM
        # ====================================================

        esm_data = (
            self.esm_embeds[
                self.names[i]
            ]
        )

        esm_emb = (
            esm_data["emb"]
            .float()
        )

        esm_mask = (
            esm_data["mask"]
            .long()
        )

        # ====================================================
        # MolFormer
        # ====================================================

        mol_feat = torch.as_tensor(
            self.mol_embeds[
                self.smiles[i]
            ],
            dtype=torch.float32
        )

        # ====================================================
        # Substrate concentration
        # ====================================================

        sm_load = torch.tensor(
            [self.sm_norm[i]],
            dtype=torch.float32
        )

        # ====================================================
        # Enzyme concentration
        # ====================================================

        enz_load = torch.tensor(
            [self.enzyme_norm[i]],
            dtype=torch.float32
        )

        return (
            esm_emb,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load
        )


# ============================================================
# 4. 模型
#
# 必须和训练代码完全一致
# ============================================================

class EnzymeSubstrateBinaryClassifier(nn.Module):

    def __init__(
        self,
        esm_dim=1280,
        mol_dim=768,
        dropout=0.1,
        hidden_dim=256
    ):

        super().__init__()

        # ====================================================
        # ESM projection
        # ====================================================

        self.esm_proj = nn.Sequential(

            nn.Linear(
                esm_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                256
            )
        )

        # ====================================================
        # MolFormer projection
        # ====================================================

        self.mol_proj = nn.Sequential(

            nn.Linear(
                mol_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                256
            )
        )

        # ====================================================
        # Transformer
        # ====================================================

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=dropout,
                batch_first=True
            )
        )

        self.esm_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
                enable_nested_tensor=False
            )
        )

        # ====================================================
        # Substrate concentration
        # ====================================================

        self.sm_proj = nn.Sequential(

            nn.Linear(
                1,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.LayerNorm(
                128
            )
        )

        # ====================================================
        # Enzyme concentration
        # ====================================================

        self.enz_proj = nn.Sequential(

            nn.Linear(
                1,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.LayerNorm(
                128
            )
        )

        # ====================================================
        # Fusion
        # ====================================================

        input_dim = (
            256
            + 256
            + 128
            + 128
        )

        # ====================================================
        # FC1
        # ====================================================

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        # ====================================================
        # FC2
        # ====================================================

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        # ====================================================
        # FC3
        # ====================================================

        self.fc3 = nn.Linear(
            hidden_dim,
            hidden_dim // 2
        )

        self.bn3 = nn.BatchNorm1d(
            hidden_dim // 2
        )

        # ====================================================
        # Dropout
        # ====================================================

        self.dropout = nn.Dropout(
            0.2
        )

        # ====================================================
        # Binary classifier
        # ====================================================

        self.classifier = nn.Linear(
            hidden_dim // 2,
            1
        )

    def forward(
        self,
        esm_feat,
        esm_mask,
        mol_feat,
        sm_load,
        enz_load
    ):

        # ====================================================
        # ESM
        # ====================================================

        esm_feat = self.esm_proj(
            esm_feat
        )

        # ====================================================
        # Transformer mask
        # ====================================================

        src_mask = (
            esm_mask == 0
        ).bool()

        # ====================================================
        # Transformer
        # ====================================================

        esm_feat = self.esm_encoder(
            esm_feat,
            src_key_padding_mask=src_mask
        )

        # ====================================================
        # Masked mean pooling
        # ====================================================

        mask_expanded = (
            esm_mask
            .unsqueeze(-1)
            .float()
        )

        denom = (
            mask_expanded
            .sum(dim=1)
            .clamp(min=1e-8)
        )

        esm_pooled = (
            esm_feat
            * mask_expanded
        ).sum(dim=1) / denom

        # ====================================================
        # MolFormer
        # ====================================================

        mol_proj = self.mol_proj(
            mol_feat
        )

        # ====================================================
        # Concentration
        # ====================================================

        sm_proj = self.sm_proj(
            sm_load
        )

        enz_proj = self.enz_proj(
            enz_load
        )

        # ====================================================
        # Multimodal Fusion
        # ====================================================

        x = torch.cat(
            [
                esm_pooled,
                enz_proj,
                mol_proj,
                sm_proj
            ],
            dim=-1
        )

        # ====================================================
        # FC1
        # ====================================================

        x = F.relu(
            self.bn1(
                self.fc1(x)
            )
        )

        # ====================================================
        # Residual
        # ====================================================

        residual = x

        x = F.relu(
            self.bn2(
                self.fc2(x)
            )
        )

        x = self.dropout(x)

        x = x + residual

        # ====================================================
        # FC3
        # ====================================================

        x = F.relu(
            self.bn3(
                self.fc3(x)
            )
        )

        # ====================================================
        # Binary output
        # ====================================================

        logits = (
            self.classifier(x)
            .squeeze(-1)
        )

        return logits


# ============================================================
# 5. 加载 scaler
# ============================================================

print("\n========== Loading scaler ==========")

with open(
    SCALER_PATH,
    "r"
) as f:

    scaler = json.load(f)


sm_mean = float(
    scaler["sm_mean"]
)

sm_std = float(
    scaler["sm_std"]
)

enz_mean = float(
    scaler["enz_mean"]
)

enz_std = float(
    scaler["enz_std"]
)

activity_threshold = float(
    scaler["activity_threshold"]
)

print("SM mean :", sm_mean)
print("SM std  :", sm_std)
print("Enz mean:", enz_mean)
print("Enz std :", enz_std)

print(
    "Activity threshold:",
    activity_threshold
)


# ============================================================
# 6. 加载 ESM
# ============================================================

print("\n========== Loading ESM features ==========")

with open(
    ESM_PKL,
    "rb"
) as f:

    esm_dict = pickle.load(f)

print(
    "ESM features:",
    len(esm_dict)
)


# ============================================================
# 7. 加载 MolFormer
# ============================================================

print("\n========== Loading MolFormer features ==========")

with open(
    MOL_PKL,
    "rb"
) as f:

    mol_dict = pickle.load(f)

print(
    "MolFormer features:",
    len(mol_dict)
)


# ============================================================
# 8. 读取预测数据
# ============================================================

print("\n========== Loading prediction CSV ==========")

df = pd.read_csv(
    INPUT_CSV
)

print(
    "Input samples:",
    len(df)
)


# ============================================================
# 9. 检查必要列
# ============================================================

required_columns = [
    "Name",
    "SM",
    "SM loading (g/L)",
    "Enzyme loading (g/L)"
]

missing_columns = [
    x for x in required_columns
    if x not in df.columns
]

if len(missing_columns) > 0:

    raise ValueError(
        "输入 CSV 缺少以下列："
        + str(missing_columns)
    )


# ============================================================
# 10. ID / SMILES
# ============================================================

df["Name"] = (
    df["Name"]
    .astype(str)
)

df["SM"] = (
    df["SM"]
    .astype(str)
)


# ============================================================
# 11. 浓度处理
#
# 必须和训练阶段完全一致：
#
# 原始 g/L
#     ↓
# fillna(0)
#     ↓
# clip(lower=0)
#     ↓
# log1p
#     ↓
# 使用训练集 mean/std 标准化
# ============================================================

SM_COL = "SM loading (g/L)"
ENZ_COL = "Enzyme loading (g/L)"


df[SM_COL] = (
    pd.to_numeric(
        df[SM_COL],
        errors="coerce"
    )
    .fillna(0)
    .clip(lower=0)
)

df[ENZ_COL] = (
    pd.to_numeric(
        df[ENZ_COL],
        errors="coerce"
    )
    .fillna(0)
    .clip(lower=0)
)


df["SM_log"] = np.log1p(
    df[SM_COL]
)

df["Enz_log"] = np.log1p(
    df[ENZ_COL]
)


# ------------------------------------------------------------
# 注意：
# 这里绝对不能重新计算 mean/std
#
# 必须使用训练阶段保存的 scaler
# ------------------------------------------------------------

df["SM_norm"] = (
    df["SM_log"] - sm_mean
) / sm_std

df["Enz_norm"] = (
    df["Enz_log"] - enz_mean
) / enz_std


# ============================================================
# 12. Dataset
# ============================================================

dataset = PredictionDataset(
    df,
    esm_dict,
    mol_dict
)


# ============================================================
# 13. DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# 14. 创建模型
# ============================================================

print("\n========== Building model ==========")

model = (
    EnzymeSubstrateBinaryClassifier()
    .to(device)
)


# ============================================================
# 15. 加载训练好的模型
# ============================================================

print("\n========== Loading trained model ==========")

ckpt = torch.load(
    MODEL_PATH,
    map_location=device
)


# ------------------------------------------------------------
# 你的训练代码保存的是：
#
# {
#     "model": model.state_dict(),
#     "threshold": best_threshold
# }
# ------------------------------------------------------------

if (
    isinstance(ckpt, dict)
    and "model" in ckpt
):

    model.load_state_dict(
        ckpt["model"]
    )

    best_threshold = float(
        ckpt["threshold"]
    )

else:

    # 如果以后保存的是纯 state_dict，
    # 则默认使用 scaler 中的阈值
    model.load_state_dict(
        ckpt
    )

    best_threshold = 0.5


print(
    "Model loaded."
)

print(
    f"Prediction threshold = "
    f"{best_threshold:.4f}"
)


# ============================================================
# 16. Prediction
# ============================================================

print("\n========== Prediction ==========")

model.eval()

probabilities = []


with torch.no_grad():

    for (
        esm_feat,
        esm_mask,
        mol_feat,
        sm_load,
        enz_load
    ) in loader:

        esm_feat = (
            esm_feat
            .to(
                device,
                non_blocking=True
            )
        )

        esm_mask = (
            esm_mask
            .to(
                device,
                non_blocking=True
            )
        )

        mol_feat = (
            mol_feat
            .to(
                device,
                non_blocking=True
            )
        )

        sm_load = (
            sm_load
            .to(
                device,
                non_blocking=True
            )
        )

        enz_load = (
            enz_load
            .to(
                device,
                non_blocking=True
            )
        )

        # ----------------------------------------------------
        # logits
        # ----------------------------------------------------

        logits = model(
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load
        )

        # ----------------------------------------------------
        # sigmoid
        # ----------------------------------------------------

        prob = torch.sigmoid(
            logits
        )

        probabilities.extend(
            prob.cpu().numpy()
        )


probabilities = np.asarray(
    probabilities
)


# ============================================================
# 17. 二分类
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# 18. 保存结果
# ============================================================

result = df.copy()


result["Pred_Active_Probability"] = (
    probabilities
)


result["Pred_Binary"] = (
    predictions
)


# ------------------------------------------------------------
# 添加文字标签
# ------------------------------------------------------------

result["Pred_Label"] = np.where(
    predictions == 1,
    "有活性",
    "无活性"
)


# ------------------------------------------------------------
# 如果输入数据本身有 Activity，
# 顺便计算真实类别
# ------------------------------------------------------------

if "Activity" in result.columns:

    result["True_Binary"] = (
        pd.to_numeric(
            result["Activity"],
            errors="coerce"
        )
        >= activity_threshold
    ).astype(int)


# ============================================================
# 19. 保存 CSV
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)

result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 输出统计
# ============================================================

n_total = len(result)

n_active = (
    predictions == 1
).sum()

n_inactive = (
    predictions == 0
).sum()


print("\n========== Prediction Result ==========")

print(
    "Total:",
    n_total
)

print(
    "Predicted inactive:",
    n_inactive,
    f"({n_inactive / n_total:.2%})"
)

print(
    "Predicted active:",
    n_active,
    f"({n_active / n_total:.2%})"
)

print(
    "Probability min:",
    probabilities.min()
)

print(
    "Probability max:",
    probabilities.max()
)

print(
    "Probability mean:",
    probabilities.mean()
)

print(
    "\nPrediction saved:"
)

print(
    OUTPUT_CSV
)
