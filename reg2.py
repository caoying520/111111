import os, json, random, pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, matthews_corrcoef, f1_score,
)

# ==================== 1. 参数 ====================
SEED, BATCH_SIZE, NUM_WORKERS = 123, 256, 4
PRETRAINED_MODEL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"
CSV_PATH  = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/13k_std_非0.csv"
ESM_PKL   = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/esm_144KRD.pkl"
MOL_PKL   = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/SM96_std.pkl"
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/split_Activity_enz_非0"
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/model_Activity_reg_enz_非0"
SM_COL, ENZ_COL, Y_COL = "SM loading (g/L)", "Enzyme loading (g/L)", "Activity"

# 渐进解冻三阶段
STAGE1_EPOCHS, STAGE1_PATIENCE = 50, 15          # 只训 fc4
STAGE2A_EPOCHS, STAGE2A_PATIENCE = 100, 20       # 训 fc1~fc4 + sm_proj/enz_proj + BN
STAGE2B_EPOCHS, STAGE2B_PATIENCE = 500, 50       # 全量微调
STAGE1_LR, STAGE2A_LR, STAGE2B_LR = 1e-3, 1e-4, 3e-5
WEIGHT_DECAY, MIN_DELTA, GRAD_CLIP = 1e-5, 1e-5, 5.0

# 非零样本损失权重（≈2.0 → 非零:零 的有效比例约 1.7:1，比原 0.86:1 更合理）
NONZERO_WEIGHT = 2.0

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 2. 工具函数 ====================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    with open(path, "rb") as f: return pickle.load(f)


def read_ids(name):
    path = os.path.join(SPLIT_DIR, name)
    if not os.path.isfile(path): raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f: return [x.strip() for x in f if x.strip()]


# ==================== 3. Dataset（带样本权重） ====================
class EnzymeDataset(Dataset):
    def __init__(self, df, esm, mol, y_mean, y_std):
        self.df, self.esm, self.mol = df.reset_index(drop=True), esm, mol
        y = np.log1p(self.df[Y_COL].to_numpy(dtype=float))
        self.y = ((y - y_mean) / y_std).astype(np.float32)
        # 非零样本加权：用原始 Y_COL > 0 判断，而非 log 后的值
        is_nonzero = (self.df[Y_COL].to_numpy(dtype=float) > 0).astype(np.float32)
        self.w = np.where(is_nonzero > 0, NONZERO_WEIGHT, 1.0).astype(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        e = self.esm[str(r["Name"])]
        emb = torch.as_tensor(np.asarray(e["emb"]), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(e["mask"]), dtype=torch.bool).reshape(-1)
        n = min(len(emb), len(mask)); emb, mask = emb[:n], mask[:n]
        if n == 0: raise ValueError(f"{r['Name']} 的 ESM 序列长度为0")
        mol = torch.as_tensor(np.asarray(self.mol[str(r["SM"])]), dtype=torch.float32).reshape(-1)
        if emb.shape[1] != 1280 or mol.numel() != 768: raise ValueError("ESM/MolFormer 特征维度错误")
        return (
            emb, mask, mol,
            torch.tensor([r["SM_norm"]],  dtype=torch.float32),
            torch.tensor([r["Enz_norm"]], dtype=torch.float32),
            torch.tensor(self.y[i]),
            str(r["idx"]), str(r["Name"]), str(r["SM"]),
            torch.tensor(self.w[i], dtype=torch.float32),   # 权重放在最后，保持 b[:6] 索引不变
        )


def collate_fn(batch):
    max_len = max(x[0].shape[0] for x in batch)
    embs, masks = [], []
    for x in batch:
        e, m = x[0], x[1]; n = min(len(e), len(m)); e, m = e[:n], m[:n]
        pad = max_len - n
        if pad:
            e = torch.cat([e, torch.zeros(pad, e.shape[1])])
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])
        embs.append(e); masks.append(m)
    return (
        torch.stack(embs), torch.stack(masks),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        torch.stack([x[5] for x in batch]),
        [x[6] for x in batch], [x[7] for x in batch], [x[8] for x in batch],
        torch.stack([x[9] for x in batch]),
    )


# ==================== 4. 模型（结构不变） ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.esm_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.sm_proj  = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128))
        self.enz_proj = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128))
        self.fc1, self.bn1 = nn.Linear(768, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2)
        self.fc4, self.dropout = nn.Linear(hidden_dim // 2, 1), nn.Dropout(0.2)

    def forward(self, esm, mask, mol, sm, enz):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1e-8)
        x = torch.cat([pooled, self.enz_proj(enz), self.mol_proj(mol), self.sm_proj(sm)], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


def load_pretrained(model, path, device):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt.pop("fc4.weight", None); ckpt.pop("fc4.bias", None)
    current = model.state_dict()
    matched = {k: v for k, v in ckpt.items() if k in current and v.shape == current[k].shape}
    ratio = sum(v.numel() for v in matched.values()) / sum(v.numel() for v in current.values())
    print(f"预训练权重匹配比例: {ratio:.4f}，匹配层数: {len(matched)}")
    if ratio < 0.80: raise RuntimeError("预训练权重匹配比例低于80%，请检查模型结构")
    model.load_state_dict(matched, strict=False)
    nn.init.xavier_uniform_(model.fc4.weight); nn.init.zeros_(model.fc4.bias)
    return model


# ==================== 5. 损失与指标 ====================
def weighted_smooth_l1(pred, target, weight, beta=1.0):
    """样本加权 SmoothL1：非零活性样本权重更高，缓解零值对梯度的稀释。"""
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return (loss * weight).mean()


def safe_corr(func, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0: return np.nan
    try: return float(func(y, p)[0])
    except Exception: return np.nan


def regression_metrics(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE":  float(mean_absolute_error(y, p)),
        "R2":   float(r2_score(y, p)),
        "Pearson":  safe_corr(pearsonr, y, p),
        "Spearman": safe_corr(spearmanr, y, p),
    }


def enrichment_factor(y_bin, y_pred, frac):
    """EF@frac: 预测 Top-frac 中真实阳性富集倍数。y_bin 为 0/1。"""
    n = len(y_bin); k = max(1, int(n * frac))
    order = np.argsort(-y_pred)[:k]
    n_pos = int(y_bin.sum())
    if n_pos == 0: return np.nan
    hits = int(y_bin[order].sum())
    return float((hits / k) / (n_pos / n))


def scan_best_threshold(y_bin, y_pred):
    """在验证集上扫描使 MCC 最大的阈值；返回 (best_mcc, best_f1, best_thr)。"""
    thresholds = np.unique(np.percentile(y_pred, np.linspace(1, 99, 50)))
    best_mcc, best_f1, best_thr = -1.0, 0.0, 0.0
    for thr in thresholds:
        yhat = (y_pred > thr).astype(int)
        if yhat.sum() == 0: continue
        mcc = matthews_corrcoef(y_bin, yhat)
        f1  = f1_score(y_bin, yhat, zero_division=0)
        if mcc > best_mcc:
            best_mcc, best_f1, best_thr = float(mcc), float(f1), float(thr)
    return best_mcc, best_f1, best_thr


def sorting_classification_metrics(y_true, y_pred, threshold=None):
    """
    以 Activity > 0 为阳性，计算排序/分类指标。
    threshold=None 时在给定数据上扫描最优 MCC 阈值（只应在验证集上这样做）；
    传入 threshold 时直接使用（应用于测试集）。
    """
    y_bin = (y_true > 0).astype(int)
    out = {"PR_AUC": np.nan, "MCC": np.nan, "F1": np.nan,
           "EF@1%": np.nan, "EF@5%": np.nan, "Threshold": np.nan}
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return out
    try:
        out["PR_AUC"] = float(average_precision_score(y_bin, y_pred))
    except Exception:
        pass
    out["EF@1%"] = enrichment_factor(y_bin, y_pred, 0.01)
    out["EF@5%"] = enrichment_factor(y_bin, y_pred, 0.05)
    if threshold is None:
        mcc, f1, thr = scan_best_threshold(y_bin, y_pred)
        out.update(MCC=mcc, F1=f1, Threshold=thr)
    else:
        yhat = (y_pred > threshold).astype(int)
        out["MCC"] = float(matthews_corrcoef(y_bin, yhat))
        out["F1"]  = float(f1_score(y_bin, yhat, zero_division=0))
        out["Threshold"] = float(threshold)
    return out


# ==================== 6. 训练与预测 ====================
def train_epoch(model, loader, optimizer, device, training=True):
    model.train(training); total, count = 0.0, 0
    for b in loader:
        e, mask, mol, sm, enz, y = [v.to(device, non_blocking=True) for v in b[:6]]
        w = b[9].to(device, non_blocking=True)
        if training and len(y) == 1: model.eval()
        pred = model(e, mask, mol, sm, enz)
        loss = weighted_smooth_l1(pred, y, w, beta=1.0)
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
            if len(y) == 1: model.train()
        total += loss.item() * len(y); count += len(y)
    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval(); ys, ps, meta = [], [], []
    for b in loader:
        e, mask, mol, sm, enz, y = [v.to(device, non_blocking=True) for v in b[:6]]
        p = model(e, mask, mol, sm, enz).cpu().numpy()
        ys.extend(y.cpu().numpy()); ps.extend(p)
        meta.extend(zip(*b[6:9]))
    y_true = np.expm1(np.asarray(ys) * y_std + y_mean)
    y_pred = np.expm1(np.asarray(ps) * y_std + y_mean)
    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["True_Activity"], result["Pred_Activity"] = y_true, y_pred
    return regression_metrics(y_true, y_pred), result


def save_model(model, path):
    tmp = path + ".tmp"; torch.save(model.state_dict(), tmp); os.replace(tmp, path)


def _set_requires_grad(model, params, value=True):
    for p in model.parameters(): p.requires_grad = False
    for p in params: p.requires_grad = value


def train_stage(model, train_loader, val_loader, device, epochs, patience, lr,
                params, y_mean, y_std, save_path):
    _set_requires_grad(model, params, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=WEIGHT_DECAY)
    best, bad = np.inf, 0
    for epoch in range(1, epochs + 1):
        tr_loss  = train_epoch(model, train_loader, optimizer, device, True)
        val_loss = train_epoch(model, val_loader,  None,      device, False)
        val_reg, val_df = predict(model, val_loader, device, y_mean, y_std)
        val_extra = sorting_classification_metrics(
            val_df["True_Activity"].values, val_df["Pred_Activity"].values)
        rmse = val_reg["RMSE"]
        print(f"Epoch {epoch:03d}/{epochs} | TrainLoss={tr_loss:.6f} | ValLoss={val_loss:.6f} "
              f"| ValRMSE={rmse:.6f} | MAE={val_reg['MAE']:.6f} | R2={val_reg['R2']:.4f} "
              f"| EF@5%={val_extra['EF@5%']:.3f} | PR_AUC={val_extra['PR_AUC']:.4f} "
              f"| MCC={val_extra['MCC']:.4f}")
        if rmse < best - MIN_DELTA:
            best, bad = rmse, 0; save_model(model, save_path); print("  >>> 保存最佳回归模型")
        else:
            bad += 1
        if bad >= patience: print("Early Stopping"); break
    if not os.path.isfile(save_path): raise RuntimeError("没有生成最佳模型")
    model.load_state_dict(torch.load(save_path, map_location=device))


# ==================== 7. 主程序 ====================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Task: Activity Regression (weighted, gradual unfreezing)")

    esm, mol = load_pickle(ESM_PKL), load_pickle(MOL_PKL)
    print("ESM features:", len(esm), "| MolFormer features:", len(mol))
    df = pd.read_csv(CSV_PATH)
    required = ["idx", "Name", "SM", Y_COL, SM_COL, ENZ_COL]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"CSV 缺少列: {missing}")
    for c in ["idx", "Name", "SM"]: df[c] = df[c].astype(str).str.strip()
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[Y_COL]).reset_index(drop=True)
    if df["idx"].duplicated().any(): raise ValueError("idx 存在重复")
    if (df[Y_COL] <= -1).any(): raise ValueError("Activity 必须大于-1才能使用log1p")

    # ---------- 负集审计 ----------
    n_zero = int((df[Y_COL] == 0).sum()); n_pos = int((df[Y_COL] > 0).sum())
    print(f"\n[负集审计] 全部样本: {len(df)} | 零值: {n_zero} ({100*n_zero/len(df):.1f}%) | "
          f"非零: {n_pos} ({100*n_pos/len(df):.1f}%)")
    nz = df.loc[df[Y_COL] > 0, Y_COL]
    if len(nz) > 0:
        print("[负集审计] 非零样本活性分位数:", nz.quantile([0.1, 0.5, 0.9]).to_dict())
    print("[提示] 请确认零值全部为实验测得的无活性；若混有未测试/低于检测限样本，建议先剔除或降权。")

    # ---------- 划分 ----------
    ids = {n: set(read_ids(n)) for n in ["train.txt", "val.txt", "test.txt"]}
    if ids["train.txt"] & ids["val.txt"] or ids["train.txt"] & ids["test.txt"] or ids["val.txt"] & ids["test.txt"]:
        raise ValueError("Train/Val/Test idx 存在交集")
    parts = {k: df[df.idx.isin(v)].copy().reset_index(drop=True)
             for k, v in [("train", ids["train.txt"]), ("val", ids["val.txt"]), ("test", ids["test.txt"])]}
    if any(x.empty for x in parts.values()): raise ValueError("Train/Val/Test 存在空数据集")

    groups = {k: set(v.Name) for k, v in parts.items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("Train/Val/Test 存在重复酶 Name")

    # ---------- 底物跨集重叠诊断 ----------
    sm_train, sm_val, sm_test = set(parts["train"].SM), set(parts["val"].SM), set(parts["test"].SM)
    print("\n[底物重叠诊断]")
    print(f"  train∩test: {len(sm_train & sm_test)} / {len(sm_test)} 测试底物 "
          f"({100*len(sm_train & sm_test)/max(len(sm_test),1):.1f}%)")
    print(f"  train∩val : {len(sm_train & sm_val)} / {len(sm_val)} 验证底物 "
          f"({100*len(sm_train & sm_val)/max(len(sm_val),1):.1f}%)")
    if len(sm_train & sm_test) / max(len(sm_test), 1) > 0.2:
        print("  [警告] 跨集底物重叠偏高，测试性能可能偏乐观；考虑按底物骨架分组进一步验证。")

    for k, d in parts.items():
        if set(d.Name) - set(esm) or set(d.SM) - set(mol):
            raise KeyError(f"{k} 缺少 ESM 或 MolFormer 特征")
        for c in [SM_COL, ENZ_COL]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
        d["SM_log"], d["Enz_log"] = np.log1p(d[SM_COL]), np.log1p(d[ENZ_COL])

    sm_mean, sm_std   = parts["train"].SM_log.mean(),  parts["train"].SM_log.std()
    enz_mean, enz_std = parts["train"].Enz_log.mean(), parts["train"].Enz_log.std()
    sm_std  = sm_std  if np.isfinite(sm_std)  and sm_std  > 1e-12 else 1.0
    enz_std = enz_std if np.isfinite(enz_std) and enz_std > 1e-12 else 1.0
    for d in parts.values():
        d["SM_norm"]  = (d.SM_log  - sm_mean)  / sm_std
        d["Enz_norm"] = (d.Enz_log - enz_mean) / enz_std

    y_log = np.log1p(parts["train"][Y_COL])
    y_mean, y_std = float(y_log.mean()), float(y_log.std())
    if not np.isfinite(y_std) or y_std < 1e-12: y_std = 1.0

    with open(os.path.join(MODEL_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"activity_mean": y_mean, "activity_std": y_std,
                   "sm_mean": sm_mean, "sm_std": sm_std,
                   "enz_mean": enz_mean, "enz_std": enz_std,
                   "transform": "log1p+zscore", "nonzero_weight": NONZERO_WEIGHT},
                  f, ensure_ascii=False, indent=2)

    print("\nSamples:", {k: len(v) for k, v in parts.items()},
          "| Enzymes:", {k: len(v) for k, v in groups.items()})
    for k, d in parts.items():
        z = int((d[Y_COL] == 0).sum())
        print(f"{k}: n={len(d)}, zeros={z} ({100*z/len(d):.1f}%), "
              f"mean={d[Y_COL].mean():.4f}, std={d[Y_COL].std():.4f}, max={d[Y_COL].max():.4f}")

    datasets = {k: EnzymeDataset(v, esm, mol, y_mean, y_std) for k, v in parts.items()}
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
              pin_memory=device.type == "cuda", collate_fn=collate_fn)
    loaders = {"train": DataLoader(datasets["train"], shuffle=True,  drop_last=False, **kw),
               "val":   DataLoader(datasets["val"],   shuffle=False, drop_last=False, **kw),
               "test":  DataLoader(datasets["test"],  shuffle=False, drop_last=False, **kw)}

    model = load_pretrained(EnzymeSubstrateRegressor().to(device), PRETRAINED_MODEL, device)

    # ---------- Stage 1: 只训 fc4 ----------
    print("\n========== STAGE 1: TRAIN FC4 ONLY ==========")
    s1_path = os.path.join(MODEL_DIR, "stage1_best.pth")
    train_stage(model, loaders["train"], loaders["val"], device,
                STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR,
                list(model.fc4.parameters()), y_mean, y_std, s1_path)

    # ---------- Stage 2a: 解冻 MLP 头 + 物理特征投影 + BN ----------
    print("\n========== STAGE 2a: UNFREEZE MLP HEADS + SM/ENZ PROJ ==========")
    s2a_params = (list(model.fc1.parameters()) + list(model.fc2.parameters()) +
                  list(model.fc3.parameters()) + list(model.fc4.parameters()) +
                  list(model.bn1.parameters()) + list(model.bn2.parameters()) +
                  list(model.bn3.parameters()) +
                  list(model.sm_proj.parameters()) + list(model.enz_proj.parameters()))
    s2a_path = os.path.join(MODEL_DIR, "stage2a_best.pth")
    train_stage(model, loaders["train"], loaders["val"], device,
                STAGE2A_EPOCHS, STAGE2A_PATIENCE, STAGE2A_LR,
                s2a_params, y_mean, y_std, s2a_path)

    # ---------- Stage 2b: 全量微调 ----------
    print("\n========== STAGE 2b: FULL FINE-TUNE ==========")
    s2b_path = os.path.join(MODEL_DIR, "best_regression.pth")
    train_stage(model, loaders["train"], loaders["val"], device,
                STAGE2B_EPOCHS, STAGE2B_PATIENCE, STAGE2B_LR,
                list(model.parameters()), y_mean, y_std, s2b_path)

    # ---------- 验证集评估（并确定分类阈值） ----------
    print("\n========== VAL: REGRESSION + SORTING/CLASSIFICATION ==========")
    val_reg, val_df = predict(model, loaders["val"], device, y_mean, y_std)
    val_extra = sorting_classification_metrics(
        val_df["True_Activity"].values, val_df["Pred_Activity"].values, threshold=None)
    for k, v in {**val_reg, **val_extra}.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")
    val_df.to_csv(os.path.join(MODEL_DIR, "val_predictions.csv"), index=False, encoding="utf-8-sig")
    chosen_thr = val_extra["Threshold"]
    print(f"[阈值] 验证集最优 MCC 阈值 = {chosen_thr:.6f}（将应用于测试集，避免测试集泄漏）")

    # ---------- 测试集评估 ----------
    print("\n========== TEST: REGRESSION + SORTING/CLASSIFICATION ==========")
    test_reg, test_df = predict(model, loaders["test"], device, y_mean, y_std)
    test_extra = sorting_classification_metrics(
        test_df["True_Activity"].values, test_df["Pred_Activity"].values, threshold=chosen_thr)
    for k, v in {**test_reg, **test_extra}.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")
    test_df.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    # 汇总
    summary = {"val": {**val_reg, **val_extra}, "test": {**test_reg, **test_extra}}
    with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nBest model:", s2b_path)
    print("Scaler:",    os.path.join(MODEL_DIR, "scaler.json"))
    print("Summary:",   os.path.join(MODEL_DIR, "metrics_summary.json"))
    print("Output:", MODEL_DIR)


if __name__ == "__main__":
    main()
