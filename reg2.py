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
SEED, BATCH_SIZE, NUM_WORKERS = 123, 256, 4

PRETRAINED_MODEL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"
CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/13k_std.csv"
ESM_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/esm_144KRD.pkl"
MOL_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/SM96_std.pkl"
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/split_Activity_enz"
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/model_Activity_reg_enz_zero_nonzero"

SM_COL, ENZ_COL, Y_COL = "SM loading (g/L)", "Enzyme loading (g/L)", "Activity"

# 浓度列为原始g/L时用raw_gL；只有当CSV已有的_norm与预训练模型完全同源时才用existing_norm
CONC_MODE = "raw_gL"  # "raw_gL" 或 "existing_norm"

# 零值/非零值权重：非零样本适度提高权重，避免零值占多数时模型过度贴近0
# 先用1.5，不建议一开始就设很高；后续可对比1.0、1.5、2.0
ZERO_WEIGHT, NONZERO_WEIGHT = 1.0, 1.5

STAGE1_EPOCHS, STAGE2_EPOCHS = 50, 500
STAGE1_PATIENCE, STAGE2_PATIENCE = 15, 50
STAGE1_LR, STAGE2_LR, WEIGHT_DECAY = 1e-3, 3e-5, 1e-5
MIN_DELTA, GRAD_CLIP = 1e-5, 5.0

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


def regression_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan,
                "Pearson": np.nan, "Spearman": np.nan}
    return {
        "N": int(len(y)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "Pearson": safe_corr(pearsonr, y, p),
        "Spearman": safe_corr(spearmanr, y, p)
    }


def print_metrics(title, metrics):
    print(title)
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


# ==================== 3. Dataset ====================
class EnzymeDataset(Dataset):
    def __init__(self, df, esm, mol, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.esm, self.mol = esm, mol
        y_raw = self.df[Y_COL].to_numpy(dtype=np.float64)
        y_log = np.log1p(y_raw)
        self.y = ((y_log - y_mean) / y_std).astype(np.float32)
        self.weight = np.where(
            y_raw > 0, NONZERO_WEIGHT, ZERO_WEIGHT
        ).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        e = self.esm[str(r["Name"])]
        emb = torch.as_tensor(np.asarray(e["emb"]), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(e["mask"]), dtype=torch.bool).reshape(-1)
        n = min(len(emb), len(mask))
        if n == 0:
            raise ValueError(f"{r['Name']} 的ESM序列长度为0")
        emb, mask = emb[:n], mask[:n]

        mol = torch.as_tensor(np.asarray(self.mol[str(r["SM"])]), dtype=torch.float32).reshape(-1)
        if emb.shape[1] != 1280 or mol.numel() != 768:
            raise ValueError(f"特征维度错误：ESM={tuple(emb.shape)}, MolFormer={mol.numel()}")

        return (
            emb, mask, mol,
            torch.tensor([r["SM_norm"]], dtype=torch.float32),
            torch.tensor([r["Enz_norm"]], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.weight[i], dtype=torch.float32),
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
        if pad:
            e = torch.cat([e, torch.zeros(pad, e.shape[1])], dim=0)
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)], dim=0)
        embs.append(e)
        masks.append(m)

    return (
        torch.stack(embs), torch.stack(masks),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        torch.stack([x[5] for x in batch]),
        torch.stack([x[6] for x in batch]),
        [x[7] for x in batch], [x[8] for x in batch], [x[9] for x in batch]
    )


# ==================== 4. 模型：沿用预训练结构 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.esm_encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.sm_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.enz_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.fc1, self.bn1 = nn.Linear(768, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2)
        self.fc4, self.dropout = nn.Linear(hidden_dim // 2, 1), nn.Dropout(0.2)

    def forward(self, esm, mask, mol, sm, enz):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1e-8)
        x = torch.cat(
            [pooled, self.enz_proj(enz), self.mol_proj(mol), self.sm_proj(sm)],
            dim=-1
        )
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
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt.pop("fc4.weight", None)
    ckpt.pop("fc4.bias", None)

    current = model.state_dict()
    matched = {k: v for k, v in ckpt.items() if k in current and v.shape == current[k].shape}
    ratio = sum(v.numel() for v in matched.values()) / sum(v.numel() for v in current.values())
    print(f"预训练权重匹配比例: {ratio:.4f}，匹配参数层数: {len(matched)}")
    if ratio < 0.80:
        raise RuntimeError("预训练权重匹配比例低于80%，请检查模型结构和权重")
    model.load_state_dict(matched, strict=False)
    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)
    return model


# ==================== 6. 训练、预测 ====================
def run_epoch(model, loader, optimizer, loss_fn, device, training, stage1=False):
    if training:
        if stage1:
            model.eval()
            model.fc4.train()
        else:
            model.train()
    else:
        model.eval()

    total, count = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for b in loader:
            e, mask, mol, sm, enz, y, weight = [
                v.to(device, non_blocking=True) for v in b[:7]
            ]
            pred = model(e, mask, mol, sm, enz)
            per_sample = loss_fn(pred, y)
            loss = (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
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
        p = model(e, mask, mol, sm, enz).cpu().numpy()
        ys.extend(y.cpu().numpy())
        ps.extend(p)
        meta.extend(zip(*b[7:10]))

    y_true = np.expm1(np.asarray(ys) * y_std + y_mean)
    y_pred = np.expm1(np.asarray(ps) * y_std + y_mean)
    y_pred = np.maximum(y_pred, 0.0)

    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["True_Activity"] = y_true
    result["Pred_Activity"] = y_pred
    return result


def evaluate_and_print(title, result):
    y = result["True_Activity"].to_numpy()
    p = result["Pred_Activity"].to_numpy()
    is_zero = (y == 0)
    is_nonzero = (y > 0)

    print_metrics(title + " ALL", regression_metrics(y, p))
    if is_zero.any():
        print_metrics(title + " ZERO subset", regression_metrics(y[is_zero], p[is_zero]))
        print(f"  ZERO预测均值: {p[is_zero].mean():.6f}")
    if is_nonzero.any():
        print_metrics(title + " NONZERO subset", regression_metrics(y[is_nonzero], p[is_nonzero]))
        print(f"  NONZERO真实均值: {y[is_nonzero].mean():.6f} | 预测均值: {p[is_nonzero].mean():.6f}")

    # log空间指标用于观察长尾压缩后的拟合情况，不作为主要选模指标
    log_rmse = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))
    print(f"{title} log1p-RMSE: {log_rmse:.6f}")


def save_model(model, path):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def train_stage(
    model, train_loader, val_loader, device, loss_fn,
    epochs, patience, lr, params, y_mean, y_std, save_path, stage1=False
):
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
        threshold=MIN_DELTA, min_lr=1e-7
    )

    best, bad = np.inf, 0
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(
            model, train_loader, optimizer, loss_fn, device, True, stage1
        )
        val_loss = run_epoch(
            model, val_loader, None, loss_fn, device, False, False
        )
        val_result = predict(model, val_loader, device, y_mean, y_std)
        val_metrics = regression_metrics(
            val_result.True_Activity.to_numpy(),
            val_result.Pred_Activity.to_numpy()
        )
        rmse = val_metrics["RMSE"]
        scheduler.step(rmse)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} | LR={current_lr:.2e} | "
            f"TrainLoss={tr_loss:.6f} | ValLoss={val_loss:.6f} | "
            f"ValRMSE={rmse:.6f} | ValMAE={val_metrics['MAE']:.6f} | "
            f"ValR2={val_metrics['R2']:.4f}"
        )

        if rmse < best - MIN_DELTA:
            best, bad = rmse, 0
            save_model(model, save_path)
            print("  >>> 保存最佳模型")
        else:
            bad += 1

        if bad >= patience:
            print("Early Stopping")
            break

    if not os.path.isfile(save_path):
        raise RuntimeError(f"没有生成最佳模型: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))


# ==================== 7. 主程序 ====================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Task: Zero + Nonzero Pure Activity Regression")
    print(f"Loss weights: zero={ZERO_WEIGHT}, nonzero={NONZERO_WEIGHT}")

    esm, mol = load_pickle(ESM_PKL), load_pickle(MOL_PKL)
    print("ESM features:", len(esm), "| MolFormer features:", len(mol))

    df = pd.read_csv(CSV_PATH)
    required = ["idx", "Name", "SM", Y_COL, SM_COL, ENZ_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV缺少列: {missing}")

    for c in ["idx", "Name", "SM"]:
        df[c] = df[c].astype(str).str.strip()
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[Y_COL]).reset_index(drop=True)

    if df["idx"].duplicated().any():
        raise ValueError("idx存在重复")
    if (df[Y_COL] < 0).any():
        raise ValueError("检测到负Activity；当前脚本假设Activity非负，请先核对标签")

    ids = {n: set(read_ids(n)) for n in ["train.txt", "val.txt", "test.txt"]}
    if (
        ids["train.txt"] & ids["val.txt"]
        or ids["train.txt"] & ids["test.txt"]
        or ids["val.txt"] & ids["test.txt"]
    ):
        raise ValueError("Train/Val/Test的idx存在交集")

    parts = {
        k: df[df.idx.isin(v)].copy().reset_index(drop=True)
        for k, v in [
            ("train", ids["train.txt"]),
            ("val", ids["val.txt"]),
            ("test", ids["test.txt"])
        ]
    }
    if any(x.empty for x in parts.values()):
        raise ValueError("Train/Val/Test存在空数据集")

    groups = {k: set(v.Name) for k, v in parts.items()}
    if (
        groups["train"] & groups["val"]
        or groups["train"] & groups["test"]
        or groups["val"] & groups["test"]
    ):
        raise ValueError("Train/Val/Test存在重复酶Name；当前脚本要求按酶隔离")

    for k, d in parts.items():
        if set(d.Name) - set(esm):
            raise KeyError(f"{k}缺少ESM特征")
        if set(d.SM) - set(mol):
            raise KeyError(f"{k}缺少MolFormer特征")

    if CONC_MODE == "existing_norm":
        for d in parts.values():
            for c in ["SM_norm", "Enz_norm"]:
                if c not in d.columns:
                    raise KeyError(f"CSV缺少{c}，不能使用existing_norm")
                d[c] = pd.to_numeric(d[c], errors="coerce")
                if d[c].isna().any():
                    raise ValueError(f"{c}存在空值")
        conc_scaler = {"mode": "existing_norm"}
    elif CONC_MODE == "raw_gL":
        for d in parts.values():
            for c in [SM_COL, ENZ_COL]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                if d[c].isna().any() or (d[c] < 0).any():
                    raise ValueError(f"{c}有空值或负值，请检查原始g/L浓度")
            d["SM_log"] = np.log1p(d[SM_COL])
            d["Enz_log"] = np.log1p(d[ENZ_COL])

        sm_mean, sm_std = float(parts["train"].SM_log.mean()), float(parts["train"].SM_log.std())
        enz_mean, enz_std = float(parts["train"].Enz_log.mean()), float(parts["train"].Enz_log.std())
        sm_std = sm_std if np.isfinite(sm_std) and sm_std > 1e-12 else 1.0
        enz_std = enz_std if np.isfinite(enz_std) and enz_std > 1e-12 else 1.0

        for d in parts.values():
            d["SM_norm"] = (d.SM_log - sm_mean) / sm_std
            d["Enz_norm"] = (d.Enz_log - enz_mean) / enz_std

        conc_scaler = {
            "mode": "raw_gL_log1p_train_zscore",
            "sm_mean": sm_mean, "sm_std": sm_std,
            "enz_mean": enz_mean, "enz_std": enz_std
        }
    else:
        raise ValueError("CONC_MODE只能是existing_norm或raw_gL")

    y_train = parts["train"][Y_COL].to_numpy(dtype=np.float64)
    y_log = np.log1p(y_train)
    y_mean, y_std = float(y_log.mean()), float(y_log.std(ddof=1))
    if not np.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0

    scaler = {
        "activity_mean": y_mean, "activity_std": y_std,
        "activity_transform": "log1p+train_zscore",
        "concentration": conc_scaler,
        "zero_weight": ZERO_WEIGHT, "nonzero_weight": NONZERO_WEIGHT
    }
    with open(os.path.join(MODEL_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    print("Samples:", {k: len(v) for k, v in parts.items()})
    print("Enzymes:", {k: len(v) for k, v in groups.items()})
    for k, d in parts.items():
        y = d[Y_COL].to_numpy()
        print(
            f"{k}: zero={(y == 0).sum()} ({(y == 0).mean():.2%}), "
            f"nonzero={(y > 0).sum()} ({(y > 0).mean():.2%}), "
            f"mean={y.mean():.4f}, median={np.median(y):.4f}, "
            f"p90={np.quantile(y, 0.90):.4f}, p99={np.quantile(y, 0.99):.4f}, "
            f"max={y.max():.4f}"
        )

    datasets = {
        k: EnzymeDataset(v, esm, mol, y_mean, y_std)
        for k, v in parts.items()
    }
    common = dict(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"), collate_fn=collate_fn
    )
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **common)
    }

    model = load_pretrained(
        EnzymeSubstrateRegressor().to(device), PRETRAINED_MODEL, device
    )
    # 逐样本SmoothL1，零值与非零值使用温和权重
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")

    print("\n========== STAGE 1: TRAIN FC4 ONLY ==========")
    stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")
    train_stage(
        model, loaders["train"], loaders["val"], device, loss_fn,
        STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR,
        list(model.fc4.parameters()), y_mean, y_std, stage1_path,
        stage1=True
    )

    print("\n========== STAGE 2: FINE-TUNE FULL MODEL ==========")
    best_path = os.path.join(MODEL_DIR, "best_regression.pth")
    train_stage(
        model, loaders["train"], loaders["val"], device, loss_fn,
        STAGE2_EPOCHS, STAGE2_PATIENCE, STAGE2_LR,
        list(model.parameters()), y_mean, y_std, best_path,
        stage1=False
    )

    print("\n========== TEST: ZERO + NONZERO REGRESSION ==========")
    test_result = predict(model, loaders["test"], device, y_mean, y_std)
    evaluate_and_print("TEST", test_result)
    test_result.to_csv(
        os.path.join(MODEL_DIR, "test_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    val_result = predict(model, loaders["val"], device, y_mean, y_std)
    evaluate_and_print("VAL", val_result)
    val_result.to_csv(
        os.path.join(MODEL_DIR, "val_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("Best model:", best_path)
    print("Scaler:", os.path.join(MODEL_DIR, "scaler.json"))
    print("Output:", MODEL_DIR)


if __name__ == "__main__":
    main()
