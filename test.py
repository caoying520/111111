
# -*- coding: utf-8 -*-
import os
import random
import pickle
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 参数
# ============================================================

SEED = 123
BATCH_SIZE = 256
NUM_WORKERS = 4

PRETRAINED_MODEL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"

CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/train_data3.csv"

ESM_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/esm_feats_emb_mask_test_6_0.pkl"

MOL_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/SM.pkl"

# 如果采用底物分组，使用 split_Activity_substrate
# 如果采用按酶分组，改成 split_Activity_enz
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/split_Activity_substrate"

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"

MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_regression"
os.makedirs(MODEL_DIR, exist_ok=True)

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 500

STAGE1_LR = 1e-3
STAGE2_LR = 3e-5
WEIGHT_DECAY = 1e-5

STAGE1_PATIENCE = 15
STAGE2_PATIENCE = 50
MIN_DELTA = 1e-5

ACTIVITY_THRESHOLD = 0.03

ID_COL = "idx"
ENZYME_COL = "Name"
SUBSTRATE_COL = "SM"
ACTIVITY_COL = "Activity"

SM_COL = "SM loading (g/L)"
ENZ_COL = "Enzyme loading (g/L)"


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 3. Dataset
# ============================================================

class EnzymeSubstrateDataset(Dataset):

    def __init__(
        self,
        df,
        esm_embeds,
        mol_embeds,
        activity_mean,
        activity_std
    ):

        self.df = df.reset_index(drop=True)

        self.names = self.df[ENZYME_COL].astype(str).tolist()

        self.smiles = (
            self.df[SUBSTRATE_COL].astype(str).str.strip().tolist()
        )

        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        activity = self.df[ACTIVITY_COL].astype(float).values

        if np.min(activity) <= -1:
            raise ValueError(
                "当前 Activity 存在 <= -1 的值，不能直接使用 log1p。"
            )

        activity_log = np.log1p(activity)

        self.labels = (
            (activity_log - activity_mean) / activity_std
        ).astype(np.float32)

        self.raw_activity = activity.astype(np.float32)

        self.sm_norm = (
            self.df["SM_norm"].astype(float).values.astype(np.float32)
        )

        # Activity 已经是比活性，不再使用酶浓度作为预测变量。
        # 保留原来的 enz_proj 结构以兼容 Conv checkpoint。
        self.enz_norm = np.zeros(
            len(self.df),
            dtype=np.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        name = self.names[i]
        smiles = self.smiles[i]

        if name not in self.esm_embeds:
            raise KeyError(f"ESM 特征不存在: {name}")

        if smiles not in self.mol_embeds:
            raise KeyError(f"MolFormer 特征不存在: {smiles}")

        esm_data = self.esm_embeds[name]

        esm_emb = torch.as_tensor(
            esm_data["emb"],
            dtype=torch.float32
        )

        esm_mask = torch.as_tensor(
            esm_data["mask"],
            dtype=torch.long
        )

        mol_feat = torch.as_tensor(
            self.mol_embeds[smiles],
            dtype=torch.float32
        )

        sm_load = torch.tensor(
            [self.sm_norm[i]],
            dtype=torch.float32
        )

        enz_load = torch.tensor(
            [self.enz_norm[i]],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[i],
            dtype=torch.float32
        )

        raw_activity = torch.tensor(
            self.raw_activity[i],
            dtype=torch.float32
        )

        return (
            esm_emb,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            label,
            raw_activity,
            name,
            smiles
        )


# ============================================================
# 4. 变长 ESM 序列 padding
# ============================================================

def collate_fn(batch):

    embs = [x[0] for x in batch]
    masks = [x[1] for x in batch]

    max_len = max(x.shape[0] for x in embs)

    emb_dim = embs[0].shape[-1]

    padded_embs = []
    padded_masks = []

    for emb, mask in zip(embs, masks):

        length = min(
            emb.shape[0],
            mask.shape[0]
        )

        emb = emb[:length]
        mask = mask[:length]

        pad_len = max_len - length

        if pad_len > 0:

            emb = torch.cat(
                [
                    emb,
                    torch.zeros(
                        pad_len,
                        emb_dim,
                        dtype=emb.dtype
                    )
                ],
                dim=0
            )

            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        pad_len,
                        dtype=mask.dtype
                    )
                ],
                dim=0
            )

        padded_embs.append(emb)
        padded_masks.append(mask)

    esm_feat = torch.stack(padded_embs)
    esm_mask = torch.stack(padded_masks)

    mol_feat = torch.stack([x[2] for x in batch])
    sm_load = torch.stack([x[3] for x in batch])
    enz_load = torch.stack([x[4] for x in batch])

    labels = torch.stack([x[5] for x in batch])
    raw_activity = torch.stack([x[6] for x in batch])

    names = [x[7] for x in batch]
    smiles = [x[8] for x in batch]

    return (
        esm_feat,
        esm_mask,
        mol_feat,
        sm_load,
        enz_load,
        labels,
        raw_activity,
        names,
        smiles
    )


# ============================================================
# 5. 模型
# 保留原 Conv 模型结构
# ============================================================

class EnzymeSubstrateRegressor(nn.Module):

    def __init__(
        self,
        esm_dim=1280,
        mol_dim=768,
        dropout=0.1,
        hidden_dim=256
    ):

        super().__init__()

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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc4 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        esm_feat,
        esm_mask,
        mol_feat,
        sm_load,
        enz_load
    ):

        esm_feat = self.esm_proj(esm_feat)

        src_mask = (esm_mask == 0)

        esm_feat = self.esm_encoder(
            esm_feat,
            src_key_padding_mask=src_mask
        )

        mask_expanded = esm_mask.unsqueeze(-1).float()

        denom = mask_expanded.sum(
            dim=1
        ).clamp(min=1e-8)

        esm_pooled = (
            (esm_feat * mask_expanded).sum(dim=1) / denom
        )

        mol_proj = self.mol_proj(mol_feat)

        sm_proj = self.sm_proj(sm_load)

        enz_proj = self.enz_proj(enz_load)

        x = torch.cat(
            [
                esm_pooled,
                enz_proj,
                mol_proj,
                sm_proj
            ],
            dim=-1
        )

        x = F.relu(self.bn1(self.fc1(x)))

        residual = x

        x = F.relu(self.bn2(self.fc2(x)))

        x = self.dropout(x)

        x = x + residual

        x = F.relu(self.bn3(self.fc3(x)))

        return self.fc4(x).squeeze(-1)


# ============================================================
# 6. 加载 Conv 预训练权重
# ============================================================

def load_pretrained(model, path, device):

    ckpt = torch.load(
        path,
        map_location=device
    )

    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    elif "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    new_ckpt = {}

    for key, value in ckpt.items():

        if key.startswith("module."):
            key = key[7:]

        if key.startswith("fc4."):
            continue

        new_ckpt[key] = value

    model_dict = model.state_dict()

    matched = {
        k: v
        for k, v in new_ckpt.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    matched_numel = sum(
        v.numel() for v in matched.values()
    )

    total_numel = sum(
        v.numel()
        for k, v in model_dict.items()
        if not k.startswith("fc4.")
    )

    missing, unexpected = model.load_state_dict(
        matched,
        strict=False
    )

    ratio = matched_numel / max(total_numel, 1)

    print("\n========== Pretrained Loading ==========")

    print("Matched tensors:", len(matched))
    print("Parameter coverage:", f"{ratio:.2%}")
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    if ratio < 0.80:
        raise RuntimeError(
            "预训练权重匹配率低于 80%，"
            "请检查当前模型和 Conv checkpoint 是否为同一架构。"
        )

    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)

    print("Conv 预训练特征层加载完成，fc4 已重新初始化。")

    return model


# ============================================================
# 7. 预测
# ============================================================

def predict(
    model,
    loader,
    device,
    activity_mean,
    activity_std
):

    model.eval()

    all_pred = []
    all_true = []
    all_names = []
    all_smiles = []

    with torch.no_grad():

        for batch in loader:

            (
                esm_feat,
                esm_mask,
                mol_feat,
                sm_load,
                enz_load,
                y,
                raw_activity,
                names,
                smiles
            ) = batch

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)

            pred = model(
                esm_feat,
                esm_mask,
                mol_feat,
                sm_load,
                enz_load
            )

            all_pred.extend(pred.cpu().numpy())
            all_true.extend(raw_activity.numpy())
            all_names.extend(names)
            all_smiles.extend(smiles)

    pred = np.asarray(all_pred)
    true = np.asarray(all_true)

    # 反标准化
    pred_log = pred * activity_std + activity_mean

    # 反 log1p
    pred_activity = np.expm1(pred_log)

    pred_activity = np.nan_to_num(
        pred_activity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        true,
        pred_activity,
        pred,
        all_names,
        all_smiles
    )


# ============================================================
# 8. 评价指标
# ============================================================

def evaluate(true, pred):

    true = np.asarray(true)
    pred = np.asarray(pred)

    rmse = np.sqrt(
        mean_squared_error(true, pred)
    )

    mae = mean_absolute_error(
        true,
        pred
    )

    r2 = r2_score(
        true,
        pred
    )

    if len(np.unique(true)) > 1 and len(np.unique(pred)) > 1:

        pearson = pearsonr(true, pred)[0]

        spearman = spearmanr(true, pred).statistic

    else:

        pearson = np.nan
        spearman = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman
    }


# ============================================================
# 9. 训练
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train_mode=True
):

    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_num = 0

    for batch in loader:

        (
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            y,
            raw_activity,
            names,
            smiles
        ) = batch

        esm_feat = esm_feat.to(device)
        esm_mask = esm_mask.to(device)
        mol_feat = mol_feat.to(device)
        sm_load = sm_load.to(device)
        enz_load = enz_load.to(device)
        y = y.to(device)

        pred = model(
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load
        )

        loss = criterion(pred, y)

        if train_mode:

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)

    return total_loss / max(total_num, 1)


# ============================================================
# 10. 读取 split
# ============================================================

def read_split(name):

    path = os.path.join(
        SPLIT_DIR,
        name
    )

    with open(path, "r") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


# ============================================================
# 11. 主程序
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # 读取 ESM 和 MolFormer
    # --------------------------------------------------------

    with open(ESM_PKL, "rb") as f:
        esm_dict = pickle.load(f)

    with open(MOL_PKL, "rb") as f:
        mol_dict = pickle.load(f)

    print(
        f"ESM={len(esm_dict)}, MolFormer={len(mol_dict)}"
    )

    # --------------------------------------------------------
    # 读取 TXT
    # --------------------------------------------------------

    train_idx = read_split(TRAIN_TXT)
    val_idx = read_split(VAL_TXT)
    test_idx = read_split(TEST_TXT)

    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))

    # --------------------------------------------------------
    # 读取 CSV
    # --------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    df[ID_COL] = df[ID_COL].astype(str)

    df[ENZYME_COL] = df[ENZYME_COL].astype(str)

    df[SUBSTRATE_COL] = (
        df[SUBSTRATE_COL].astype(str).str.strip()
    )

    df[ACTIVITY_COL] = pd.to_numeric(
        df[ACTIVITY_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            ID_COL,
            ENZYME_COL,
            SUBSTRATE_COL,
            ACTIVITY_COL
        ]
    ).reset_index(drop=True)

    if df[ID_COL].duplicated().any():
        raise ValueError("CSV 中 idx 不唯一，请先检查数据。")

    df_train = df[
        df[ID_COL].isin(train_idx)
    ].copy().reset_index(drop=True)

    df_val = df[
        df[ID_COL].isin(val_idx)
    ].copy().reset_index(drop=True)

    df_test = df[
        df[ID_COL].isin(test_idx)
    ].copy().reset_index(drop=True)

    if min(len(df_train), len(df_val), len(df_test)) == 0:
        raise ValueError("train/val/test 存在空集，请检查 TXT。")

    print("\n========== Data Split ==========")

    for name, d in [
        ("Train", df_train),
        ("Val", df_val),
        ("Test", df_test)
    ]:

        print(
            f"{name}: rows={len(d)}, "
            f"enzymes={d[ENZYME_COL].nunique()}, "
            f"substrates={d[SUBSTRATE_COL].nunique()}, "
            f"zero={(d[ACTIVITY_COL] == 0).sum()}"
        )

    # --------------------------------------------------------
    # 检查特征
    # --------------------------------------------------------

    for name, d in [
        ("Train", df_train),
        ("Val", df_val),
        ("Test", df_test)
    ]:

        missing_esm = (
            set(d[ENZYME_COL]) - set(esm_dict.keys())
        )

        missing_mol = (
            set(d[SUBSTRATE_COL]) - set(mol_dict.keys())
        )

        if missing_esm:
            raise KeyError(
                f"{name} 缺少 ESM 特征: {list(missing_esm)[:5]}"
            )

        if missing_mol:
            raise KeyError(
                f"{name} 缺少 MolFormer 特征: {list(missing_mol)[:5]}"
            )

    # --------------------------------------------------------
    # 浓度处理
    # --------------------------------------------------------

    for d in [df_train, df_val, df_test]:

        d[SM_COL] = pd.to_numeric(
            d[SM_COL],
            errors="coerce"
        )

        d[ENZ_COL] = pd.to_numeric(
            d[ENZ_COL],
            errors="coerce"
        )

        if d[SM_COL].isna().any():
            raise ValueError("底物浓度存在缺失值。")

        if d[ENZ_COL].isna().any():
            raise ValueError("酶浓度存在缺失值。")

        d["SM_log"] = np.log1p(
            d[SM_COL].clip(lower=0)
        )

        d["Enz_log"] = np.log1p(
            d[ENZ_COL].clip(lower=0)
        )

    # 只用训练集拟合浓度 scaler
    sm_mean = df_train["SM_log"].mean()
    sm_std = max(df_train["SM_log"].std(), 1e-12)

    for d in [df_train, df_val, df_test]:

        d["SM_norm"] = (
            d["SM_log"] - sm_mean
        ) / sm_std

        # 保留 Enz_log 供记录检查，但模型输入固定为 0

    # --------------------------------------------------------
    # Activity 标签处理
    # --------------------------------------------------------

    if (df[ACTIVITY_COL] <= -1).any():
        raise ValueError(
            "Activity 存在 <= -1 的值，log1p 无法处理。"
        )

    train_log = np.log1p(
        df_train[ACTIVITY_COL].values
    )

    activity_mean = train_log.mean()

    activity_std = max(
        train_log.std(),
        1e-12
    )

    print("\n========== Activity ==========")

    print("Activity=0:", (df[ACTIVITY_COL] == 0).sum())

    print("Activity min:", df[ACTIVITY_COL].min())

    print("Activity max:", df[ACTIVITY_COL].max())

    print("log1p mean:", activity_mean)

    print("log1p std:", activity_std)

    pd.Series({
        "activity_mean": activity_mean,
        "activity_std": activity_std,
        "sm_mean": sm_mean,
        "sm_std": sm_std,
        "transform": "log1p+zscore"
    }).to_json(
        os.path.join(
            MODEL_DIR,
            "activity_scaler.json"
        )
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_set = EnzymeSubstrateDataset(
        df_train,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    val_set = EnzymeSubstrateDataset(
        df_val,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    test_set = EnzymeSubstrateDataset(
        df_test,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    model = EnzymeSubstrateRegressor().to(device)

    model = load_pretrained(
        model,
        PRETRAINED_MODEL,
        device
    )

    criterion = nn.SmoothL1Loss(beta=1.0)

    # --------------------------------------------------------
    # Stage 1：只训练 fc4
    # --------------------------------------------------------

    print("\n========== Stage 1 ==========")

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc4.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.fc4.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY
    )

    best_rmse = np.inf
    bad_epochs = 0

    stage1_path = os.path.join(
        MODEL_DIR,
        "stage1_best.pth"
    )

    for epoch in range(STAGE1_EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=False
        )

        true_val, pred_val, _, _, _ = predict(
            model,
            val_loader,
            device,
            activity_mean,
            activity_std
        )

        val_metrics = evaluate(
            true_val,
            pred_val
        )

        print(
            f"Stage1 {epoch + 1:03d}/{STAGE1_EPOCHS} | "
            f"Loss={train_loss:.5f} | "
            f"RMSE={val_metrics['RMSE']:.5f} | "
            f"MAE={val_metrics['MAE']:.5f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"Pearson={val_metrics['Pearson']:.4f}"
        )

        if val_metrics["RMSE"] < best_rmse - MIN_DELTA:

            best_rmse = val_metrics["RMSE"]
            bad_epochs = 0

            torch.save(
                model.state_dict(),
                stage1_path
            )

            print("  >>> Stage1 best model saved")

        else:

            bad_epochs += 1

        if bad_epochs >= STAGE1_PATIENCE:

            print("Stage1 early stopping")

            break

    model.load_state_dict(
        torch.load(
            stage1_path,
            map_location=device
        )
    )

    # --------------------------------------------------------
    # Stage 2：全模型低学习率微调
    # --------------------------------------------------------

    print("\n========== Stage 2 ==========")

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY
    )

    best_rmse = np.inf
    bad_epochs = 0

    best_path = os.path.join(
        MODEL_DIR,
        "best_model.pth"
    )

    for epoch in range(STAGE2_EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=True
        )

        true_val, pred_val, _, _, _ = predict(
            model,
            val_loader,
            device,
            activity_mean,
            activity_std
        )

        val_metrics = evaluate(
            true_val,
            pred_val
        )

        print(
            f"Stage2 {epoch + 1:03d}/{STAGE2_EPOCHS} | "
            f"Loss={train_loss:.5f} | "
            f"RMSE={val_metrics['RMSE']:.5f} | "
            f"MAE={val_metrics['MAE']:.5f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"Pearson={val_metrics['Pearson']:.4f} | "
            f"Spearman={val_metrics['Spearman']:.4f}"
        )

        if val_metrics["RMSE"] < best_rmse - MIN_DELTA:

            best_rmse = val_metrics["RMSE"]
            bad_epochs = 0

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                f"  >>> Best model saved: "
                f"RMSE={best_rmse:.5f}"
            )

        else:

            bad_epochs += 1

        if bad_epochs >= STAGE2_PATIENCE:

            print(
                f"Stage2 early stopping: "
                f"{STAGE2_PATIENCE} epochs without improvement"
            )

            break

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device
        )
    )

    true_test, pred_test, _, names, smiles = predict(
        model,
        test_loader,
        device,
        activity_mean,
        activity_std
    )

    test_metrics = evaluate(
        true_test,
        pred_test
    )

    print("\n========== FINAL TEST ==========")

    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # --------------------------------------------------------
    # 保存测试集预测
    # --------------------------------------------------------

    result = df_test.copy()

    result["True_Activity"] = true_test

    result["Pred_Activity"] = pred_test

    result["Residual"] = (
        result["Pred_Activity"] - result["True_Activity"]
    )

    result["True_Binary"] = (
        result["True_Activity"] >= ACTIVITY_THRESHOLD
    ).astype(int)

    result["Pred_Binary"] = (
        result["Pred_Activity"] >= ACTIVITY_THRESHOLD
    ).astype(int)

    result.to_csv(
        os.path.join(
            MODEL_DIR,
            "test_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        os.path.join(
            MODEL_DIR,
            "test_metrics.csv"
        ),
        index=False
    )

    print("\n结果保存位置:", MODEL_DIR)


if __name__ == "__main__":
    main()
