
# -*- coding: utf-8 -*-
import os
import random
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/split_Activity_enz2"

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"

MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_rank_reg"

os.makedirs(MODEL_DIR, exist_ok=True)

ACTIVITY_THRESHOLD = 0.03

RANK_WEIGHT = 0.1

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 500

STAGE1_PATIENCE = 15
STAGE2_PATIENCE = 50

MIN_DELTA = 1e-5

SM_COL = "SM loading (g/L)"
ENZ_COL = "Enzyme loading (g/L)"


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed=123):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
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

        self.idx = self.df["idx"].astype(str).tolist()

        self.names = self.df["Name"].astype(str).tolist()

        self.smiles = self.df["SM"].astype(str).tolist()

        activity = self.df["Activity"].astype(float).values

        if np.min(activity) <= -1:
            raise ValueError("log1p(Activity) 要求所有 Activity > -1")

        activity_log = np.log1p(activity)

        self.labels = (
            (activity_log - activity_mean) / activity_std
        ).astype(np.float32)

        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        self.sm_norm = self.df["SM_norm"].astype(float).values

        self.enz_norm = self.df["Enz_norm"].astype(float).values

    def __len__(self):

        return len(self.df)

    def __getitem__(self, i):

        name = self.names[i]

        smiles = self.smiles[i]

        esm_data = self.esm_embeds[name]

        esm_emb = torch.as_tensor(
            esm_data["emb"],
            dtype=torch.float32
        )

        esm_mask = torch.as_tensor(
            esm_data["mask"],
            dtype=torch.long
        )

        L = min(esm_emb.shape[0], esm_mask.shape[0])

        esm_emb = esm_emb[:L]
        esm_mask = esm_mask[:L]

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

        y = torch.tensor(
            self.labels[i],
            dtype=torch.float32
        )

        return (
            esm_emb,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            y,
            smiles,
            name,
            self.idx[i]
        )


# ============================================================
# 4. 动态 padding
# ============================================================

def collate_fn(batch):

    max_len = max(x[0].shape[0] for x in batch)

    embs = []
    masks = []

    for x in batch:

        emb = x[0]
        mask = x[1]

        L = min(emb.shape[0], mask.shape[0])

        emb = emb[:L]
        mask = mask[:L]

        pad_len = max_len - L

        if pad_len > 0:

            emb = torch.cat(
                [
                    emb,
                    torch.zeros(
                        pad_len,
                        emb.shape[1],
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

        embs.append(emb)
        masks.append(mask)

    return (
        torch.stack(embs),
        torch.stack(masks),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        torch.stack([x[5] for x in batch]),
        [x[6] for x in batch],
        [x[7] for x in batch],
        [x[8] for x in batch]
    )


# ============================================================
# 5. 模型
# 保留原始模型主体
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

        self.fc1 = nn.Linear(768, hidden_dim)
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

        denom = mask_expanded.sum(dim=1) + 1e-8

        esm_pooled = (
            esm_feat * mask_expanded
        ).sum(dim=1) / denom

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

        x = self.fc4(x)

        return x.squeeze(-1)


# ============================================================
# 6. 加载 Conv 预训练权重
# ============================================================

def load_pretrained(model, path, device):

    ckpt = torch.load(
        path,
        map_location=device
    )

    if isinstance(ckpt, dict) and "state_dict" in ckpt:

        ckpt = ckpt["state_dict"]

    ckpt = {
        k.replace("module.", ""): v
        for k, v in ckpt.items()
    }

    ckpt.pop("fc4.weight", None)
    ckpt.pop("fc4.bias", None)

    model_dict = model.state_dict()

    compatible = {
        k: v
        for k, v in ckpt.items()
        if k in model_dict
        and model_dict[k].shape == v.shape
    }

    total_params = sum(
        v.numel() for v in model_dict.values()
    )

    matched_params = sum(
        model_dict[k].numel() for k in compatible
    )

    ratio = matched_params / total_params

    print("\n预训练权重加载检查")
    print("匹配参数比例:", round(ratio, 4))
    print("匹配参数数量:", len(compatible))

    if ratio < 0.80:

        raise RuntimeError(
            "预训练权重匹配比例低于80%，"
            "请检查模型结构与原始 Conv 模型是否一致。"
        )

    missing, unexpected = model.load_state_dict(
        compatible,
        strict=False
    )

    print("Missing:", missing)
    print("Unexpected:", unexpected)

    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)

    return model


# ============================================================
# 7. Pairwise Ranking Loss
# 只比较相同底物下不同酶
# ============================================================

def pairwise_rank_loss(pred, y, smiles, names):

    device = pred.device

    B = len(y)

    same_sm = torch.tensor(
        [
            [smiles[i] == smiles[j] for j in range(B)]
            for i in range(B)
        ],
        dtype=torch.bool,
        device=device
    )

    different_enzyme = torch.tensor(
        [
            [names[i] != names[j] for j in range(B)]
            for i in range(B)
        ],
        dtype=torch.bool,
        device=device
    )

    upper = torch.triu(
        torch.ones(
            B,
            B,
            dtype=torch.bool,
            device=device
        ),
        diagonal=1
    )

    true_diff = y[:, None] - y[None, :]

    pred_diff = pred[:, None] - pred[None, :]

    valid = (
        same_sm
        & different_enzyme
        & upper
        & (torch.abs(true_diff) > 1e-6)
    )

    if not valid.any():

        return pred.sum() * 0.0

    sign = torch.sign(true_diff[valid])

    rank_loss = F.softplus(
        -sign * pred_diff[valid]
    ).mean()

    return rank_loss


# ============================================================
# 8. 回归指标
# ============================================================

def regression_metrics(true, pred):

    rmse = np.sqrt(
        mean_squared_error(true, pred)
    )

    mae = mean_absolute_error(true, pred)

    r2 = r2_score(true, pred)

    if np.std(true) > 0 and np.std(pred) > 0:

        pearson = pearsonr(true, pred)[0]

        spearman = spearmanr(true, pred)[0]

    else:

        pearson = np.nan
        spearman = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Spearman_Global": spearman
    }


# ============================================================
# 9. 同一底物内部排序评价
# ============================================================

def ranking_metrics(df):

    spearman_list = []

    correct_pairs = 0
    total_pairs = 0

    for sm, group in df.groupby("SM"):

        group = group.drop_duplicates(
            subset=["Name"],
            keep="first"
        )

        if len(group) < 2:
            continue

        true = group["True_Activity"].values

        pred = group["Pred_Activity"].values

        if len(np.unique(true)) > 1 and len(np.unique(pred)) > 1:

            rho = spearmanr(true, pred)[0]

            if np.isfinite(rho):
                spearman_list.append(rho)

        for i in range(len(group)):

            for j in range(i + 1, len(group)):

                true_diff = true[i] - true[j]

                if abs(true_diff) <= 1e-8:
                    continue

                pred_diff = pred[i] - pred[j]

                total_pairs += 1

                if abs(pred_diff) <= 1e-12:

                    correct_pairs += 0.5

                elif np.sign(true_diff) == np.sign(pred_diff):

                    correct_pairs += 1

    pair_acc = (
        correct_pairs / total_pairs
        if total_pairs > 0 else np.nan
    )

    mean_spearman = (
        np.mean(spearman_list)
        if spearman_list else np.nan
    )

    return {
        "Pairwise_Accuracy": pair_acc,
        "Mean_Substrate_Spearman": mean_spearman,
        "Rank_Substrates": len(spearman_list),
        "Valid_Pairs": total_pairs
    }


# ============================================================
# 10. 训练一个 Epoch
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
        model.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_rank = 0.0
    total_count = 0

    for batch in loader:

        (
            esm_feat,
            esm_mask,
            mol_feat,
            sm_load,
            enz_load,
            y,
            smiles,
            names,
            idx
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

        reg_loss = criterion(pred, y)

        rank_loss = pairwise_rank_loss(
            pred,
            y,
            smiles,
            names
        )

        loss = reg_loss + RANK_WEIGHT * rank_loss

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        n = y.size(0)

        total_loss += loss.item() * n
        total_reg += reg_loss.item() * n
        total_rank += rank_loss.item() * n

        total_count += n

    return {
        "Loss": total_loss / max(total_count, 1),
        "RegLoss": total_reg / max(total_count, 1),
        "RankLoss": total_rank / max(total_count, 1)
    }


# ============================================================
# 11. 验证与预测
# ============================================================

def evaluate(
    model,
    loader,
    device,
    criterion,
    activity_mean,
    activity_std
):

    model.eval()

    all_pred = []
    all_true = []

    all_sm = []
    all_names = []
    all_idx = []

    total_loss = 0.0
    total_reg = 0.0
    total_rank = 0.0
    total_count = 0

    with torch.no_grad():

        for batch in loader:

            (
                esm_feat,
                esm_mask,
                mol_feat,
                sm_load,
                enz_load,
                y,
                smiles,
                names,
                idx
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

            reg_loss = criterion(pred, y)

            rank_loss = pairwise_rank_loss(
                pred,
                y,
                smiles,
                names
            )

            loss = reg_loss + RANK_WEIGHT * rank_loss

            n = y.size(0)

            total_loss += loss.item() * n
            total_reg += reg_loss.item() * n
            total_rank += rank_loss.item() * n
            total_count += n

            all_pred.extend(pred.cpu().numpy())

            all_true.extend(y.cpu().numpy())

            all_sm.extend(smiles)

            all_names.extend(names)

            all_idx.extend(idx)

    all_pred = np.asarray(all_pred)

    all_true = np.asarray(all_true)

    # 反标准化
    pred_activity = np.expm1(
        all_pred * activity_std + activity_mean
    )

    true_activity = np.expm1(
        all_true * activity_std + activity_mean
    )

    pred_activity = np.where(
        np.isfinite(pred_activity),
        pred_activity,
        0
    )

    result = pd.DataFrame({

        "idx": all_idx,
        "Name": all_names,
        "SM": all_sm,
        "True_Activity": true_activity,
        "Pred_Activity": pred_activity

    })

    reg = regression_metrics(
        true_activity,
        pred_activity
    )

    rank = ranking_metrics(result)

    metrics = {

        "Loss": total_loss / max(total_count, 1),

        "RegLoss": total_reg / max(total_count, 1),

        "RankLoss": total_rank / max(total_count, 1),

        **reg,

        **rank
    }

    return metrics, result


# ============================================================
# 12. 按底物输出预测排名
# ============================================================

def save_rank_result(result, path):

    result = result.copy()

    result["Pred_Rank"] = (
        result.groupby("SM")["Pred_Activity"]
        .rank(ascending=False, method="min")
    )

    result["True_Rank"] = (
        result.groupby("SM")["True_Activity"]
        .rank(ascending=False, method="min")
    )

    result = result.sort_values(
        ["SM", "Pred_Rank"]
    )

    result.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 13. 主程序
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # 读取特征
    # --------------------------------------------------------

    with open(ESM_PKL, "rb") as f:
        esm_dict = pickle.load(f)

    with open(MOL_PKL, "rb") as f:
        mol_dict = pickle.load(f)

    print("ESM:", len(esm_dict))
    print("MolFormer:", len(mol_dict))

    # --------------------------------------------------------
    # 读取划分
    # --------------------------------------------------------

    def read_split(name):

        with open(os.path.join(SPLIT_DIR, name)) as f:

            return [
                x.strip()
                for x in f
                if x.strip()
            ]

    train_idx = read_split(TRAIN_TXT)

    val_idx = read_split(VAL_TXT)

    test_idx = read_split(TEST_TXT)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    df["idx"] = df["idx"].astype(str)

    df["Name"] = df["Name"].astype(str)

    df["SM"] = df["SM"].astype(str)

    df["Activity"] = pd.to_numeric(
        df["Activity"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Activity"]
    ).reset_index(drop=True)

    if df["idx"].duplicated().any():

        raise ValueError("idx 必须是唯一记录编号")

    df_train = df[
        df["idx"].isin(train_idx)
    ].reset_index(drop=True)

    df_val = df[
        df["idx"].isin(val_idx)
    ].reset_index(drop=True)

    df_test = df[
        df["idx"].isin(test_idx)
    ].reset_index(drop=True)

    if min(len(df_train), len(df_val), len(df_test)) == 0:

        raise ValueError("Train/Val/Test 存在空数据集")

    # --------------------------------------------------------
    # 检查按酶划分
    # --------------------------------------------------------

    g_train = set(df_train["Name"])
    g_val = set(df_val["Name"])
    g_test = set(df_test["Name"])

    assert not (g_train & g_val)
    assert not (g_train & g_test)
    assert not (g_val & g_test)

    print("\n酶分组检查通过")

    print("Train enzymes:", len(g_train))
    print("Val enzymes:", len(g_val))
    print("Test enzymes:", len(g_test))

    print("Train rows:", len(df_train))
    print("Val rows:", len(df_val))
    print("Test rows:", len(df_test))

    print(
        "Train-Val common substrates:",
        len(set(df_train.SM) & set(df_val.SM))
    )

    print(
        "Train-Test common substrates:",
        len(set(df_train.SM) & set(df_test.SM))
    )

    # --------------------------------------------------------
    # 浓度标准化
    # 只用训练集计算 mean/std
    # --------------------------------------------------------

    for d in [df_train, df_val, df_test]:

        d[SM_COL] = pd.to_numeric(
            d[SM_COL],
            errors="coerce"
        ).fillna(0).clip(lower=0)

        d[ENZ_COL] = pd.to_numeric(
            d[ENZ_COL],
            errors="coerce"
        ).fillna(0).clip(lower=0)

        d["SM_log"] = np.log1p(d[SM_COL])

        d["Enz_log"] = np.log1p(d[ENZ_COL])

    sm_mean = df_train["SM_log"].mean()

    sm_std = df_train["SM_log"].std() + 1e-12

    enz_mean = df_train["Enz_log"].mean()

    enz_std = df_train["Enz_log"].std() + 1e-12

    for d in [df_train, df_val, df_test]:

        d["SM_norm"] = (
            d["SM_log"] - sm_mean
        ) / sm_std

        d["Enz_norm"] = (
            d["Enz_log"] - enz_mean
        ) / enz_std

    # --------------------------------------------------------
    # Activity 标签标准化
    # --------------------------------------------------------

    if df["Activity"].min() <= -1:

        raise ValueError("Activity <= -1，不能使用 log1p")

    train_log = np.log1p(
        df_train["Activity"].values
    )

    activity_mean = train_log.mean()

    activity_std = train_log.std() + 1e-12

    print("\nActivity mean:", activity_mean)

    print("Activity std:", activity_std)

    scaler = {

        "activity_mean": float(activity_mean),

        "activity_std": float(activity_std),

        "sm_mean": float(sm_mean),

        "sm_std": float(sm_std),

        "enz_mean": float(enz_mean),

        "enz_std": float(enz_std),

        "transform": "log1p+zscore",

        "rank_weight": RANK_WEIGHT

    }

    pd.Series(scaler).to_json(
        os.path.join(MODEL_DIR, "scaler.json")
    )

    # --------------------------------------------------------
    # Dataset
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
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
    # Stage 1
    # 只训练 fc4
    # --------------------------------------------------------

    print("\n========== Stage 1 ==========")

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc4.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.fc4.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    best_rmse = np.inf
    bad_epochs = 0

    stage1_path = os.path.join(
        MODEL_DIR,
        "stage1_best.pth"
    )

    for epoch in range(STAGE1_EPOCHS):

        train_m = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=False
        )

        val_m, _ = evaluate(
            model,
            val_loader,
            device,
            criterion,
            activity_mean,
            activity_std
        )

        print(
            f"Stage1 {epoch+1:03d}/{STAGE1_EPOCHS} | "
            f"TrainLoss={train_m['Loss']:.5f} | "
            f"TrainReg={train_m['RegLoss']:.5f} | "
            f"TrainRank={train_m['RankLoss']:.5f} | "
            f"ValLoss={val_m['Loss']:.5f} | "
            f"ValRMSE={val_m['RMSE']:.5f} | "
            f"ValMAE={val_m['MAE']:.5f} | "
            f"ValR2={val_m['R2']:.4f} | "
            f"ValPairAcc={val_m['Pairwise_Accuracy']:.4f} | "
            f"ValSpearman={val_m['Mean_Substrate_Spearman']:.4f}"
        )

        if val_m["RMSE"] < best_rmse - MIN_DELTA:

            best_rmse = val_m["RMSE"]

            bad_epochs = 0

            torch.save(
                model.state_dict(),
                stage1_path
            )

        else:

            bad_epochs += 1

        if bad_epochs >= STAGE1_PATIENCE:

            print("Stage1 Early Stopping")

            break

    model.load_state_dict(
        torch.load(stage1_path, map_location=device)
    )

    # --------------------------------------------------------
    # Stage 2
    # 全模型低学习率微调
    # --------------------------------------------------------

    print("\n========== Stage 2 ==========")

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-5
    )

    best_rmse = np.inf
    best_pair_acc = -np.inf
    best_spearman = -np.inf

    bad_epochs = 0

    best_reg_path = os.path.join(
        MODEL_DIR,
        "best_regression.pth"
    )

    best_rank_path = os.path.join(
        MODEL_DIR,
        "best_ranking.pth"
    )

    for epoch in range(STAGE2_EPOCHS):

        train_m = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_mode=True
        )

        val_m, _ = evaluate(
            model,
            val_loader,
            device,
            criterion,
            activity_mean,
            activity_std
        )

        print(
            f"Stage2 {epoch+1:03d}/{STAGE2_EPOCHS} | "
            f"TrainLoss={train_m['Loss']:.5f} | "
            f"TrainReg={train_m['RegLoss']:.5f} | "
            f"TrainRank={train_m['RankLoss']:.5f} | "
            f"ValLoss={val_m['Loss']:.5f} | "
            f"ValRMSE={val_m['RMSE']:.5f} | "
            f"ValMAE={val_m['MAE']:.5f} | "
            f"ValR2={val_m['R2']:.4f} | "
            f"ValPearson={val_m['Pearson']:.4f} | "
            f"ValPairAcc={val_m['Pairwise_Accuracy']:.4f} | "
            f"ValSpearman={val_m['Mean_Substrate_Spearman']:.4f}"
        )

        # 最佳回归模型
        if val_m["RMSE"] < best_rmse - MIN_DELTA:

            best_rmse = val_m["RMSE"]

            bad_epochs = 0

            torch.save(
                model.state_dict(),
                best_reg_path
            )

            print("  >>> 保存最佳回归模型")

        else:

            bad_epochs += 1

        # 最佳排序模型
        pair_acc = val_m["Pairwise_Accuracy"]

        mean_sp = val_m["Mean_Substrate_Spearman"]

        if np.isfinite(pair_acc):

            if (
                pair_acc > best_pair_acc
                or (
                    np.isclose(pair_acc, best_pair_acc)
                    and np.isfinite(mean_sp)
                    and mean_sp > best_spearman
                )
            ):

                best_pair_acc = pair_acc

                best_spearman = (
                    mean_sp if np.isfinite(mean_sp)
                    else best_spearman
                )

                torch.save(
                    model.state_dict(),
                    best_rank_path
                )

                print("  >>> 保存最佳排序模型")

        if bad_epochs >= STAGE2_PATIENCE:

            print("Stage2 Early Stopping")

            break

    # --------------------------------------------------------
    # Test：回归最佳模型
    # --------------------------------------------------------

    print("\n========== TEST: BEST REGRESSION ==========")

    model.load_state_dict(
        torch.load(best_reg_path, map_location=device)
    )

    test_reg, result_reg = evaluate(
        model,
        test_loader,
        device,
        criterion,
        activity_mean,
        activity_std
    )

    for k, v in test_reg.items():
        print(k, ":", v)

    save_rank_result(
        result_reg,
        os.path.join(
            MODEL_DIR,
            "test_predictions_regression.csv"
        )
    )

    # --------------------------------------------------------
    # Test：排序最佳模型
    # --------------------------------------------------------

    if os.path.exists(best_rank_path):

        print("\n========== TEST: BEST RANKING ==========")

        model.load_state_dict(
            torch.load(best_rank_path, map_location=device)
        )

        test_rank, result_rank = evaluate(
            model,
            test_loader,
            device,
            criterion,
            activity_mean,
            activity_std
        )

        for k, v in test_rank.items():
            print(k, ":", v)

        save_rank_result(
            result_rank,
            os.path.join(
                MODEL_DIR,
                "test_predictions_ranking.csv"
            )
        )

    print("\n全部训练完成！")

    print("输出目录:", MODEL_DIR)


if __name__ == "__main__":

    main()
