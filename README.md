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
# 1. 参数
# ============================================================

SEED = 123

BATCH_SIZE = 256

# ============================================================
# 原来已经训练好的回归模型
# ============================================================

PRETRAINED_MODEL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "enzyme_train/model/reg/"
    "model_resnet_krd_conc_S2_KRD_M_55%_add/"
    "best_model_add.pth"
)

# ============================================================
# 新 Activity 数据
# ============================================================

CSV_PATH = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/train_data3.csv"
)

# ============================================================
# ESM 特征
# ============================================================

ESM_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/esm_feats_emb_mask_test_6_0.pkl"
)

# ============================================================
# MolFormer 特征
# ============================================================

MOL_PKL = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/SM.pkl"
)

# ============================================================
# 数据划分
# ============================================================

SPLIT_DIR = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/splits_Activity"
)

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"

# ============================================================
# 输出目录
# ============================================================

MODEL_DIR = (
    "/data/caoying/softwares/酶R2/enzyme_交接20260609/"
    "微调/model_Activity_binary"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ============================================================
# Activity 二分类阈值
# ============================================================

ACTIVITY_THRESHOLD = 0.03

# ============================================================
# Stage 1
# ============================================================

STAGE1_EPOCHS = 50
STAGE1_LR = 1e-3

# ============================================================
# Stage 2
# ============================================================

STAGE2_EPOCHS = 500
STAGE2_LR = 3e-5

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

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        self.idx = (
            self.df["Name"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # SMILES
        # ----------------------------------------------------

        self.smiles = (
            self.df["SM"]
            .tolist()
        )

        # ----------------------------------------------------
        # Activity
        # ----------------------------------------------------

        activity = (
            self.df["Activity"]
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # 二分类标签
        #
        # Activity < 0.03  -> 0
        # Activity >= 0.03 -> 1
        # ----------------------------------------------------

        self.labels = (
            activity >= ACTIVITY_THRESHOLD
        ).astype(np.float32)

        # ----------------------------------------------------
        # 特征
        # ----------------------------------------------------

        self.esm_embeds = esm_embeds

        self.mol_embeds = mol_embeds

        # ----------------------------------------------------
        # 浓度
        # ----------------------------------------------------

        if (
            "SM_norm" not in self.df.columns
            or
            "Enz_norm" not in self.df.columns
        ):

            raise ValueError(
                "DataFrame 必须包含 "
                "'SM_norm' 和 'Enz_norm'"
            )

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
            x for x in self.idx
            if x not in self.esm_embeds
        ]

        if len(missing_esm) > 0:

            raise KeyError(
                f"ESM.pkl 中缺少 "
                f"{len(missing_esm)} 个 Name。"
                f"例如：{missing_esm[:5]}"
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
                f"SM.pkl 中缺少 "
                f"{len(missing_mol)} 个 SM。"
                f"例如：{missing_mol[:5]}"
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, i):

        # ====================================================
        # ESM
        # ====================================================

        idx = self.idx[i]

        esm_data = (
            self.esm_embeds[idx]
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

        # ====================================================
        # Binary label
        # ====================================================

        label = torch.tensor(
            self.labels[i],
            dtype=torch.float32
        )

        return (
            esm_emb,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            label
        )


# ============================================================
# 4. 模型
# ============================================================

class EnzymeSubstrateBinaryClassifier(
    nn.Module
):

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
        # FC
        # ====================================================

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        self.fc3 = nn.Linear(
            hidden_dim,
            hidden_dim // 2
        )

        self.bn3 = nn.BatchNorm1d(
            hidden_dim // 2
        )

        self.dropout = nn.Dropout(
            0.2
        )

        # ====================================================
        # 新的二分类头
        #
        # 输出 logits
        #
        # 不在这里 sigmoid
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
# 5. 加载原始回归模型
# ============================================================

def load_pretrained(
    model,
    path,
    device
):

    print(
        "\n========== Loading pretrained model =========="
    )

    ckpt = torch.load(
        path,
        map_location=device
    )

    # --------------------------------------------------------
    # 如果 checkpoint 是：
    #
    # {"state_dict": ...}
    # --------------------------------------------------------

    if (
        isinstance(ckpt, dict)
        and
        "state_dict" in ckpt
    ):

        ckpt = ckpt["state_dict"]

    # --------------------------------------------------------
    # 去掉 DataParallel 的 module.
    # --------------------------------------------------------

    new_ckpt = {}

    for k, v in ckpt.items():

        if k.startswith(
            "module."
        ):

            k = k[7:]

        new_ckpt[k] = v

    ckpt = new_ckpt

    # --------------------------------------------------------
    # 原模型 fc4 是回归头
    #
    # 新模型 classifier 是二分类头
    #
    # 因此原 fc4 不加载
    # --------------------------------------------------------

    ckpt.pop(
        "fc4.weight",
        None
    )

    ckpt.pop(
        "fc4.bias",
        None
    )

    # --------------------------------------------------------
    # 加载 backbone
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            ckpt,
            strict=False
        )
    )

    print(
        "Missing keys:"
    )

    for x in missing:
        print(
            "  ",
            x
        )

    print(
        "Unexpected keys:"
    )

    for x in unexpected:
        print(
            "  ",
            x
        )

    # --------------------------------------------------------
    # 初始化新的 classifier
    # --------------------------------------------------------

    nn.init.xavier_uniform_(
        model.classifier.weight
    )

    nn.init.zeros_(
        model.classifier.bias
    )

    print(
        "\n✅ 原模型 backbone 加载完成"
    )

    print(
        "✅ 原 fc4 回归头未加载"
    )

    print(
        "✅ 新 classifier 二分类头已初始化"
    )

    return model


# ============================================================
# 6. 训练一个 epoch
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train_mode=True
):

    if train_mode:

        model.train()

    else:

        # Stage 1：
        # 冻结 backbone 时保持 BN / Dropout 状态
        model.eval()

    total_loss = 0.0

    total_samples = 0

    for (
        esm_feat,
        esm_mask,
        mol_feat,
        sm_load,
        enz_load,
        label
    ) in loader:

        # ----------------------------------------------------
        # device
        # ----------------------------------------------------

        esm_feat = esm_feat.to(
            device,
            non_blocking=True
        )

        esm_mask = esm_mask.to(
            device,
            non_blocking=True
        )

        mol_feat = mol_feat.to(
            device,
            non_blocking=True
        )

        sm_load = sm_load.to(
            device,
            non_blocking=True
        )

        enz_load = enz_load.to(
            device,
            non_blocking=True
        )

        label = label.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # forward
        # ----------------------------------------------------

        logits = model(
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load
        )

        # ----------------------------------------------------
        # BCE
        # ----------------------------------------------------

        loss = criterion(
            logits,
            label
        )

        # ----------------------------------------------------
        # backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        batch_size = (
            label.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# 7. Prediction
# ============================================================

def predict(
    model,
    loader,
    device
):

    model.eval()

    probabilities = []

    labels = []

    with torch.no_grad():

        for (
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            label
        ) in loader:

            esm_feat = esm_feat.to(
                device
            )

            esm_mask = esm_mask.to(
                device
            )

            mol_feat = mol_feat.to(
                device
            )

            sm_load = sm_load.to(
                device
            )

            enz_load = enz_load.to(
                device
            )

            # ------------------------------------------------
            # logits
            # ------------------------------------------------

            logits = model(
                esm_feat,
                esm_mask,
                mol_feat,
                sm_load,
                enz_load
            )

            # ------------------------------------------------
            # sigmoid
            # ------------------------------------------------

            prob = torch.sigmoid(
                logits
            )

            probabilities.extend(
                prob.cpu().numpy()
            )

            labels.extend(
                label.numpy()
            )

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return (
        labels,
        probabilities,
        predictions
    )


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate(
    true,
    probabilities,
    predictions
):

    accuracy = accuracy_score(
        true,
        predictions
    )

    precision = precision_score(
        true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    if len(
        np.unique(true)
    ) == 2:

        auc = roc_auc_score(
            true,
            probabilities
        )

    else:

        auc = np.nan

    # --------------------------------------------------------
    # confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true,
        predictions,
        labels=[0, 1]
    )

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

    # ========================================================
    # seed
    # ========================================================

    set_seed(
        SEED
    )

    # ========================================================
    # device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # ========================================================
    # ESM
    # ========================================================

    print(
        "\nLoading ESM features..."
    )

    with open(
        ESM_PKL,
        "rb"
    ) as f:

        esm_dict = pickle.load(
            f
        )

    # ========================================================
    # MolFormer
    # ========================================================

    print(
        "Loading MolFormer features..."
    )

    with open(
        MOL_PKL,
        "rb"
    ) as f:

        mol_dict = pickle.load(
            f
        )

    print(
        f"ESM={len(esm_dict)}"
    )

    print(
        f"MolFormer={len(mol_dict)}"
    )

    # ========================================================
    # split
    # ========================================================

    def read_split(
        filename
    ):

        path = os.path.join(
            SPLIT_DIR,
            filename
        )

        with open(
            path
        ) as f:

            return [
                x.strip()
                for x in f
                if x.strip()
            ]

    train_idx = set(
        read_split(
            TRAIN_TXT
        )
    )

    val_idx = set(
        read_split(
            VAL_TXT
        )
    )

    test_idx = set(
        read_split(
            TEST_TXT
        )
    )

    print(
        "\nSplit:"
    )

    print(
        "Train:",
        len(train_idx)
    )

    print(
        "Val:",
        len(val_idx)
    )

    print(
        "Test:",
        len(test_idx)
    )

    # ========================================================
    # CSV
    # ========================================================

    print(
        "\nLoading CSV..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    # ========================================================
    # 检查列
    # ========================================================

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

        raise ValueError(
            "CSV 缺少以下列："
            + str(
                missing_columns
            )
        )

    # ========================================================
    # ID
    # ========================================================

    df["idx"] = (
        df["idx"]
        .astype(str)
    )

    df["Name"] = (
        df["Name"]
        .astype(str)
    )

    # ========================================================
    # Activity
    # ========================================================

    df["Activity"] = pd.to_numeric(
        df["Activity"],
        errors="coerce"
    )

    raw_size = len(df)

    missing_activity = (
        df["Activity"]
        .isna()
        .sum()
    )

    print(
        "\nRaw data:",
        raw_size
    )

    print(
        "Activity NaN:",
        missing_activity
    )

    # 删除 Activity 缺失
    df = df.dropna(
        subset=["Activity"]
    ).reset_index(
        drop=True
    )

    print(
        "Valid data:",
        len(df)
    )

    # ========================================================
    # 根据 split 分数据
    # ========================================================

    df_train = (
        df[
            df["idx"]
            .isin(train_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_val = (
        df[
            df["idx"]
            .isin(val_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_test = (
        df[
            df["idx"]
            .isin(test_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\nDataset size:"
    )

    print(
        "Train:",
        len(df_train)
    )

    print(
        "Val:",
        len(df_val)
    )

    print(
        "Test:",
        len(df_test)
    )

    # ========================================================
    # 检查二分类比例
    # ========================================================

    print(
        "\n========== Binary distribution =========="
    )

    for name, d in [
        ("Train", df_train),
        ("Val", df_val),
        ("Test", df_test)
    ]:

        y = (
            d["Activity"]
            >= ACTIVITY_THRESHOLD
        ).astype(int)

        n0 = (
            y == 0
        ).sum()

        n1 = (
            y == 1
        ).sum()

        print(
            f"{name}:"
        )

        print(
            f"  0 = {n0} "
            f"({n0 / len(y):.2%})"
        )

        print(
            f"  1 = {n1} "
            f"({n1 / len(y):.2%})"
        )

    # ========================================================
    # 浓度
    # ========================================================

    SM_COL = (
        "SM loading (g/L)"
    )

    ENZ_COL = (
        "Enzyme loading (g/L)"
    )

    # ========================================================
    # 原模型的浓度处理方式
    # log1p
    # ========================================================

    for d in [
        df_train,
        df_val,
        df_test
    ]:

        d[SM_COL] = pd.to_numeric(
            d[SM_COL],
            errors="coerce"
        ).fillna(
            0
        ).clip(
            lower=0
        )

        d[ENZ_COL] = pd.to_numeric(
            d[ENZ_COL],
            errors="coerce"
        ).fillna(
            0
        ).clip(
            lower=0
        )

        d["SM_log"] = np.log1p(
            d[SM_COL]
        )

        d["Enz_log"] = np.log1p(
            d[ENZ_COL]
        )

    # ========================================================
    # 只用 train 计算 scaler
    # ========================================================

    sm_mean = (
        df_train["SM_log"]
        .mean()
    )

    sm_std = (
        df_train["SM_log"]
        .std()
        + 1e-12
    )

    enz_mean = (
        df_train["Enz_log"]
        .mean()
    )

    enz_std = (
        df_train["Enz_log"]
        .std()
        + 1e-12
    )

    # ========================================================
    # z-score
    # ========================================================

    for d in [
        df_train,
        df_val,
        df_test
    ]:

        d["SM_norm"] = (
            d["SM_log"]
            - sm_mean
        ) / sm_std

        d["Enz_norm"] = (
            d["Enz_log"]
            - enz_mean
        ) / enz_std

    # ========================================================
    # 保存 scaler
    #
    # 二分类模型实际上不需要 Activity scaler
    # 这里只保存浓度 scaler 和 threshold
    # ========================================================

    scaler = {

        "sm_mean":
            float(sm_mean),

        "sm_std":
            float(sm_std),

        "enz_mean":
            float(enz_mean),

        "enz_std":
            float(enz_std),

        "activity_threshold":
            float(ACTIVITY_THRESHOLD)
    }

    pd.Series(
        scaler
    ).to_json(
        os.path.join(
            MODEL_DIR,
            "scaler.json"
        )
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_set = (
        EnzymeSubstrateDataset(
            df_train,
            esm_dict,
            mol_dict
        )
    )

    val_set = (
        EnzymeSubstrateDataset(
            df_val,
            esm_dict,
            mol_dict
        )
    )

    test_set = (
        EnzymeSubstrateDataset(
            df_test,
            esm_dict,
            mol_dict
        )
    )

    # ========================================================
    # DataLoader
    # ========================================================

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
    # Model
    # ========================================================

    model = (
        EnzymeSubstrateBinaryClassifier()
        .to(device)
    )

    # ========================================================
    # 加载原来的回归模型
    # ========================================================

    model = load_pretrained(
        model,
        PRETRAINED_MODEL,
        device
    )

    # ========================================================
    # 计算 pos_weight
    # ========================================================

    train_y = (
        df_train["Activity"]
        >= ACTIVITY_THRESHOLD
    ).astype(int)

    positive = (
        train_y == 1
    ).sum()

    negative = (
        train_y == 0
    ).sum()

    if positive == 0:

        raise ValueError(
            "训练集没有正样本。"
        )

    pos_weight_value = (
        negative / positive
    )

    print(
        "\n========== Class weight =========="
    )

    print(
        "Negative:",
        negative
    )

    print(
        "Positive:",
        positive
    )

    print(
        "pos_weight:",
        pos_weight_value
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight_value,
                dtype=torch.float32,
                device=device
            )
        )
    )

    # ========================================================
    # Stage 1
    #
    # 只训练新的 classifier
    # ========================================================

    print(
        "\n========== Stage 1 =========="
    )

    # 冻结全部
    for p in model.parameters():

        p.requires_grad = False

    # 只解冻 classifier
    for p in model.classifier.parameters():

        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -np.inf

    stage1_path = os.path.join(
        MODEL_DIR,
        "stage1_best.pth"
    )

    for epoch in range(
        STAGE1_EPOCHS
    ):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=False
        )

        (
            true_val,
            prob_val,
            pred_val
        ) = predict(
            model,
            val_loader,
            device
        )

        m = evaluate(
            true_val,
            prob_val,
            pred_val
        )

        print(
            f"Stage1 "
            f"{epoch + 1:03d}/"
            f"{STAGE1_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"F1={m['F1']:.4f} | "
            f"Precision={m['Precision']:.4f} | "
            f"Recall={m['Recall']:.4f} | "
            f"AUC={m['AUC']:.4f}"
        )

        # ----------------------------------------------------
        # 保存 F1 最优
        # ----------------------------------------------------

        if m["F1"] > best_f1:

            best_f1 = m["F1"]

            torch.save(
                model.state_dict(),
                stage1_path
            )

    print(
        "\nStage1 best F1:",
        best_f1
    )

    # ========================================================
    # 加载 Stage1 最优
    # ========================================================

    model.load_state_dict(
        torch.load(
            stage1_path,
            map_location=device
        )
    )

    # ========================================================
    # Stage 2
    #
    # 全模型微调
    # ========================================================

    print(
        "\n========== Stage 2 =========="
    )

    # 解冻全部
    for p in model.parameters():

        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -np.inf

    best_path = os.path.join(
        MODEL_DIR,
        "best_model.pth"
    )

    for epoch in range(
        STAGE2_EPOCHS
    ):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=True
        )

        (
            true_val,
            prob_val,
            pred_val
        ) = predict(
            model,
            val_loader,
            device
        )

        m = evaluate(
            true_val,
            prob_val,
            pred_val
        )

        print(
            f"Stage2 "
            f"{epoch + 1:03d}/"
            f"{STAGE2_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"F1={m['F1']:.4f} | "
            f"Precision={m['Precision']:.4f} | "
            f"Recall={m['Recall']:.4f} | "
            f"AUC={m['AUC']:.4f}"
        )

        # ----------------------------------------------------
        # F1 最优保存
        # ----------------------------------------------------

        if m["F1"] > best_f1:

            best_f1 = m["F1"]

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                f"  >>> Best model "
                f"F1={best_f1:.4f}"
            )

    # ========================================================
    # Test
    # ========================================================

    print(
        "\n========== FINAL TEST =========="
    )

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device
        )
    )

    (
        true_test,
        prob_test,
        pred_test
    ) = predict(
        model,
        test_loader,
        device
    )

    test_m = evaluate(
        true_test,
        prob_test,
        pred_test
    )

    # ========================================================
    # 输出测试结果
    # ========================================================

    for k, v in test_m.items():

        print(
            f"{k}: {v}"
        )

    # ========================================================
    # 保存测试预测
    # ========================================================

    result = (
        df_test
        .copy()
    )

    # 原始 Activity
    result["True_Activity"] = (
        result["Activity"]
    )

    # 根据 0.03 得到真实类别
    result["True_Binary"] = (
        result["Activity"]
        >= ACTIVITY_THRESHOLD
    ).astype(int)

    # 模型输出概率
    result[
        "Pred_Active_Probability"
    ] = prob_test

    # 最终二分类
    result[
        "Pred_Binary"
    ] = pred_test

    # ========================================================
    # 保存
    # ========================================================

    output_csv = os.path.join(
        MODEL_DIR,
        "test_predictions.csv"
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n预测结果已保存："
    )

    print(
        output_csv
    )


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    main()
