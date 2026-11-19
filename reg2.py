
# -*- coding: utf-8 -*-
import os, json, random, pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==================== 1. 参数 ====================
SEED, BATCH_SIZE, NUM_WORKERS = 123, 128, 4

PRETRAINED_MODEL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"
CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/13k_std_非0.csv"
ESM_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/esm_144KRD.pkl"
MOL_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/SM96_std.pkl"
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/split_Activity_enz_非0"
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/model_Activity_reg_enz_非0"

SM_COL, ENZ_COL, Y_COL = "SM loading (g/L)", "Enzyme loading (g/L)", "Activity"

# 浓度输入必须是原始 g/L；如果 CSV 中已经是标准化值，不能直接使用当前预处理
CONCENTRATION_INPUT_MODE = "raw_gL"

STAGE1_EPOCHS, STAGE2_EPOCHS = 40, 150
STAGE1_PATIENCE, STAGE2_PATIENCE = 12, 15
STAGE1_LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MIN_DELTA = 1e-5

# 第二阶段分组学习率
LR_ESM = 1e-6
LR_TRANSFORMER = 3e-6
LR_MOL = 1e-6
LR_CONC = 1e-5
LR_FUSION = 1e-5
LR_HEAD = 3e-5

os.makedirs(MODEL_DIR, exist_ok=True)


# ==================== 2. 工具函数 ====================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def read_ids(name):
    path = os.path.join(SPLIT_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def safe_corr(func, y, p):
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return np.nan
    try:
        return float(func(y, p)[0])
    except Exception:
        return np.nan


def save_model(model, path):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


# ==================== 3. Dataset ====================
class EnzymeDataset(Dataset):
    def __init__(self, df, esm, mol, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.esm, self.mol = esm, mol
        y = np.log1p(self.df[Y_COL].to_numpy(dtype=np.float32))
        self.y = ((y - y_mean) / y_std).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        e = self.esm[str(r["Name"])]

        emb = torch.as_tensor(np.asarray(e["emb"]), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(e["mask"]), dtype=torch.bool).reshape(-1)

        n = min(len(emb), len(mask))
        emb, mask = emb[:n], mask[:n]

        if n == 0:
            raise ValueError(f"{r['Name']} 的 ESM 序列长度为0")

        mol = torch.as_tensor(
            np.asarray(self.mol[str(r["SM"])]),
            dtype=torch.float32
        ).reshape(-1)

        if emb.ndim != 2 or emb.shape[1] != 1280:
            raise ValueError(f"ESM 特征维度错误: {emb.shape}")

        if mol.numel() != 768:
            raise ValueError(f"MolFormer 特征维度错误: {mol.shape}")

        return (
            emb, mask, mol,
            torch.tensor([r["SM_norm"]], dtype=torch.float32),
            torch.tensor([r["Enz_norm"]], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            str(r["idx"]), str(r["Name"]), str(r["SM"])
        )


def collate_fn(batch):
    max_len = max(x[0].shape[0] for x in batch)
    embs, masks = [], []

    for x in batch:
        e, m = x[0], x[1]
        n = min(len(e), len(m))
        e, m = e[:n], m[:n]
        pad = max_len - n

        if pad > 0:
            e = torch.cat([e, torch.zeros(pad, e.shape[1])], dim=0)
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)], dim=0)

        embs.append(e)
        masks.append(m)

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


# ==================== 4. 模型 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
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

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.esm_encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
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

    def forward(self, esm, mask, mol, sm, enz):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)

        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-8)

        x = torch.cat([
            pooled,
            self.enz_proj(enz),
            self.mol_proj(mol),
            self.sm_proj(sm)
        ], dim=-1)

        x = F.relu(self.bn1(self.fc1(x)))
        residual = x

        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = F.relu(self.bn3(self.fc3(x)))

        return self.fc4(x).squeeze(-1)


# ==================== 5. 加载预训练权重 ====================
def load_pretrained(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = {
        k.replace("module.", ""): v
        for k, v in ckpt.items()
    }

    # 回归头重新初始化
    ckpt.pop("fc4.weight", None)
    ckpt.pop("fc4.bias", None)

    current = model.state_dict()
    matched = {
        k: v for k, v in ckpt.items()
        if k in current and v.shape == current[k].shape
    }

    ratio = (
        sum(v.numel() for v in matched.values()) /
        sum(v.numel() for v in current.values())
    )

    print(f"预训练权重匹配比例: {ratio:.4f}，匹配层数: {len(matched)}")

    if ratio < 0.80:
        raise RuntimeError("预训练权重匹配比例低于80%，请检查模型结构")

    model.load_state_dict(matched, strict=False)

    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)

    return model


# ==================== 6. 回归指标 ====================
def regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.maximum(np.asarray(p, dtype=float), 0.0)

    result = {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "Pearson": safe_corr(pearsonr, y, p),
        "Spearman": safe_corr(spearmanr, y, p),
        "LogRMSE": float(np.sqrt(np.mean(
            (np.log1p(y) - np.log1p(p)) ** 2
        )))
    }

    # 非零活性样本
    pos = y > 0
    if pos.sum() > 1:
        result["Positive_MAE"] = float(mean_absolute_error(y[pos], p[pos]))
        result["Positive_RMSE"] = float(
            np.sqrt(mean_squared_error(y[pos], p[pos]))
        )
    else:
        result["Positive_MAE"] = np.nan
        result["Positive_RMSE"] = np.nan

    # Activity >= 1
    high1 = y >= 1
    if high1.sum() > 1:
        result["Activity_ge_1_MAE"] = float(
            mean_absolute_error(y[high1], p[high1])
        )
    else:
        result["Activity_ge_1_MAE"] = np.nan

    # Activity >= 5
    high5 = y >= 5
    if high5.sum() > 1:
        result["Activity_ge_5_MAE"] = float(
            mean_absolute_error(y[high5], p[high5])
        )
    else:
        result["Activity_ge_5_MAE"] = np.nan

    return result


# ==================== 7. 训练与预测 ====================
def train_epoch(model, loader, optimizer, loss_fn, device, training=True,
                freeze_backbone=False):

    # 第一阶段使用 eval：冻结 Dropout 和 BatchNorm 运行统计量
    if training and freeze_backbone:
        model.eval()
    else:
        model.train(training)

    total, count = 0.0, 0

    for b in loader:
        e, mask, mol, sm, enz, y = [
            v.to(device, non_blocking=True) for v in b[:6]
        ]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(e, mask, mol, sm, enz)
            loss = loss_fn(pred, y)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP
                )
                optimizer.step()

        total += loss.item() * len(y)
        count += len(y)

    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval()
    ys, ps, meta = [], [], []

    for b in loader:
        e, mask, mol, sm, enz, y = [
            v.to(device, non_blocking=True) for v in b[:6]
        ]

        p = model(e, mask, mol, sm, enz)

        ys.extend(y.cpu().numpy())
        ps.extend(p.cpu().numpy())
        meta.extend(zip(*b[6:9]))

    y_true = np.expm1(np.asarray(ys) * y_std + y_mean)
    y_pred = np.expm1(np.asarray(ps) * y_std + y_mean)

    # Activity 不允许为负值
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, 0.0)

    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["True_Activity"] = y_true
    result["Pred_Activity"] = y_pred

    return regression_metrics(y_true, y_pred), result


# ==================== 8. 学习率分组 ====================
def get_stage2_optimizer(model):
    groups = [
        {"params": model.esm_proj.parameters(), "lr": LR_ESM},
        {"params": model.esm_encoder.parameters(), "lr": LR_TRANSFORMER},
        {"params": model.mol_proj.parameters(), "lr": LR_MOL},
        {"params": model.sm_proj.parameters(), "lr": LR_CONC},
        {"params": model.enz_proj.parameters(), "lr": LR_CONC},
        {
            "params": list(model.fc1.parameters()) +
                      list(model.bn1.parameters()) +
                      list(model.fc2.parameters()) +
                      list(model.bn2.parameters()) +
                      list(model.fc3.parameters()) +
                      list(model.bn3.parameters()),
            "lr": LR_FUSION
        },
        {"params": model.fc4.parameters(), "lr": LR_HEAD}
    ]

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY
    )


# ==================== 9. 训练阶段 ====================
def train_stage(
    model, train_loader, val_loader, device, loss_fn,
    epochs, patience, stage, y_mean, y_std, save_path
):
    for p in model.parameters():
        p.requires_grad = False

    if stage == 1:
        for p in model.fc4.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.fc4.parameters(),
            lr=STAGE1_LR,
            weight_decay=WEIGHT_DECAY
        )

    else:
        for p in model.parameters():
            p.requires_grad = True

        optimizer = get_stage2_optimizer(model)

    best, bad = np.inf, 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            training=True,
            freeze_backbone=(stage == 1)
        )

        val_loss = train_epoch(
            model, val_loader, None, loss_fn, device,
            training=False
        )

        val_metrics, _ = predict(
            model, val_loader, device, y_mean, y_std
        )

        score = val_metrics["LogRMSE"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"TrainLoss={tr_loss:.6f} | "
            f"ValLoss={val_loss:.6f} | "
            f"LogRMSE={score:.6f} | "
            f"RMSE={val_metrics['RMSE']:.6f} | "
            f"MAE={val_metrics['MAE']:.6f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"Pearson={val_metrics['Pearson']:.4f} | "
            f"Spearman={val_metrics['Spearman']:.4f}"
        )

        if score < best - MIN_DELTA:
            best = score
            bad = 0
            save_model(model, save_path)
            print("  >>> 保存最佳模型")
        else:
            bad += 1

        if bad >= patience:
            print("Early Stopping")
            break

    if not os.path.isfile(save_path):
        raise RuntimeError("没有生成最佳模型")

    model.load_state_dict(torch.load(save_path, map_location=device))


# ==================== 10. 数据预处理 ====================
def prepare_concentration(parts):
    if CONCENTRATION_INPUT_MODE != "raw_gL":
        raise ValueError(
            "当前脚本只接受原始 g/L 浓度。"
            "如果 CSV 中的浓度已经标准化，请先恢复原始 g/L，"
            "或重新确认预处理方式。"
        )

    for split, d in parts.items():
        for c in [SM_COL, ENZ_COL]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

            if d[c].isna().any():
                raise ValueError(f"{split} 的 {c} 存在缺失或非数字")

            if (d[c] < 0).any():
                raise ValueError(
                    f"{split} 的 {c} 出现负数。"
                    "这通常意味着输入可能已经标准化，不能直接 log1p。"
                )

        d["SM_log"] = np.log1p(d[SM_COL].astype(float))
        d["Enz_log"] = np.log1p(d[ENZ_COL].astype(float))

    sm_mean = float(parts["train"]["SM_log"].mean())
    sm_std = float(parts["train"]["SM_log"].std())
    enz_mean = float(parts["train"]["Enz_log"].mean())
    enz_std = float(parts["train"]["Enz_log"].std())

    sm_std = sm_std if np.isfinite(sm_std) and sm_std > 1e-12 else 1.0
    enz_std = enz_std if np.isfinite(enz_std) and enz_std > 1e-12 else 1.0

    for d in parts.values():
        d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
        d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std

    return sm_mean, sm_std, enz_mean, enz_std


# ==================== 11. 主程序 ====================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Task: Pure Activity Regression")
    print("Ranking Loss: OFF | Classification Loss: OFF")

    esm = load_pickle(ESM_PKL)
    mol = load_pickle(MOL_PKL)

    print("ESM features:", len(esm), "| MolFormer features:", len(mol))

    df = pd.read_csv(CSV_PATH)

    required = ["idx", "Name", "SM", Y_COL, SM_COL, ENZ_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少列: {missing}")

    for c in ["idx", "Name", "SM"]:
        df[c] = df[c].astype(str).str.strip()

    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

    if df[Y_COL].isna().any():
        raise ValueError("Activity 存在缺失或非数字，请先检查数据")

    if (df[Y_COL] < 0).any():
        raise ValueError("当前数据存在负 Activity，与本脚本的非负标签设定不一致")

    if df["idx"].duplicated().any():
        raise ValueError("idx 存在重复")

    # 读取已有划分文件
    split_names = ["train.txt", "val.txt", "test.txt"]
    ids = {n: set(read_ids(n)) for n in split_names}

    if (
        ids["train.txt"] & ids["val.txt"] or
        ids["train.txt"] & ids["test.txt"] or
        ids["val.txt"] & ids["test.txt"]
    ):
        raise ValueError("Train/Val/Test idx 存在交集")

    all_ids = set(df["idx"])
    split_ids = set.union(*ids.values())

    missing_ids = split_ids - all_ids
    if missing_ids:
        raise ValueError(
            f"划分文件中有 {len(missing_ids)} 个 idx 不存在于 CSV"
        )

    unused_ids = all_ids - split_ids
    if unused_ids:
        print(f"警告：CSV 中有 {len(unused_ids)} 条记录未进入任何划分")

    parts = {
        "train": df[df.idx.isin(ids["train.txt"])].copy().reset_index(drop=True),
        "val": df[df.idx.isin(ids["val.txt"])].copy().reset_index(drop=True),
        "test": df[df.idx.isin(ids["test.txt"])].copy().reset_index(drop=True)
    }

    if any(x.empty for x in parts.values()):
        raise ValueError("Train/Val/Test 存在空数据集")

    # 检查酶是否严格隔离
    groups = {k: set(v.Name) for k, v in parts.items()}

    if (
        groups["train"] & groups["val"] or
        groups["train"] & groups["test"] or
        groups["val"] & groups["test"]
    ):
        raise ValueError("Train/Val/Test 存在重复酶 Name")

    # 检查特征
    for k, d in parts.items():
        missing_esm = set(d.Name) - set(esm)
        missing_mol = set(d.SM) - set(mol)

        if missing_esm:
            raise KeyError(f"{k} 缺少 ESM 特征: {list(missing_esm)[:5]}")

        if missing_mol:
            raise KeyError(f"{k} 缺少 MolFormer 特征: {list(missing_mol)[:5]}")

    # 浓度预处理
    sm_mean, sm_std, enz_mean, enz_std = prepare_concentration(parts)

    # Activity 仅用训练集计算 scaler
    y_log = np.log1p(parts["train"][Y_COL].to_numpy(dtype=float))
    y_mean = float(y_log.mean())
    y_std = float(y_log.std())

    if not np.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0

    scaler = {
        "activity_mean": y_mean,
        "activity_std": y_std,
        "sm_mean": sm_mean,
        "sm_std": sm_std,
        "enz_mean": enz_mean,
        "enz_std": enz_std,
        "transform": "log1p+zscore",
        "concentration_input": "raw_gL"
    }

    with open(
        os.path.join(MODEL_DIR, "scaler.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    print("\n========== DATA SUMMARY ==========")

    print(
        "Samples:",
        {k: len(v) for k, v in parts.items()},
        "| Enzymes:",
        {k: len(v) for k, v in groups.items()}
    )

    for k, d in parts.items():
        print(
            f"{k}: zero={(d[Y_COL] == 0).sum()}, "
            f"zero_ratio={(d[Y_COL] == 0).mean():.4f}, "
            f"mean={d[Y_COL].mean():.4f}, "
            f"std={d[Y_COL].std():.4f}, "
            f"max={d[Y_COL].max():.4f}"
        )

    print("\nConcentration scaler:")
    print("SM mean/std:", sm_mean, sm_std)
    print("Enz mean/std:", enz_mean, enz_std)
    print("Activity log mean/std:", y_mean, y_std)

    # Dataset / DataLoader
    datasets = {
        k: EnzymeDataset(v, esm, mol, y_mean, y_std)
        for k, v in parts.items()
    }

    kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0)
    )

    loaders = {
        "train": DataLoader(
            datasets["train"], shuffle=True, drop_last=False, **kw
        ),
        "val": DataLoader(
            datasets["val"], shuffle=False, drop_last=False, **kw
        ),
        "test": DataLoader(
            datasets["test"], shuffle=False, drop_last=False, **kw
        )
    }

    # 模型与预训练权重
    model = EnzymeSubstrateRegressor().to(device)
    model = load_pretrained(model, PRETRAINED_MODEL, device)

    # 第一版先使用不加权 SmoothL1，保留所有零活性样本
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # Stage 1：仅训练新的回归头
    print("\n========== STAGE 1: TRAIN FC4 ==========")

    stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")

    train_stage(
        model, loaders["train"], loaders["val"],
        device, loss_fn,
        STAGE1_EPOCHS, STAGE1_PATIENCE,
        stage=1,
        y_mean=y_mean, y_std=y_std,
        save_path=stage1_path
    )

    # Stage 2：分组学习率微调整个模型
    print("\n========== STAGE 2: FINE-TUNE FULL MODEL ==========")

    best_path = os.path.join(MODEL_DIR, "best_regression.pth")

    train_stage(
        model, loaders["train"], loaders["val"],
        device, loss_fn,
        STAGE2_EPOCHS, STAGE2_PATIENCE,
        stage=2,
        y_mean=y_mean, y_std=y_std,
        save_path=best_path
    )

    # 最终测试集评估
    print("\n========== TEST: PURE REGRESSION ==========")

    test_metrics, test_result = predict(
        model, loaders["test"], device, y_mean, y_std
    )

    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    test_result.to_csv(
        os.path.join(MODEL_DIR, "test_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 验证集预测
    val_metrics, val_result = predict(
        model, loaders["val"], device, y_mean, y_std
    )

    val_result.to_csv(
        os.path.join(MODEL_DIR, "val_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        os.path.join(MODEL_DIR, "test_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\nBest model:", best_path)
    print("Scaler:", os.path.join(MODEL_DIR, "scaler.json"))
    print("Output:", MODEL_DIR)


if __name__ == "__main__":
    main()
