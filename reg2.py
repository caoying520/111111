
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
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/split_Activity_enz_clean"
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/model_Activity_reg_enz/split2"
Y_COL = "Activity"
STAGE1_EPOCHS, STAGE2_EPOCHS = 50, 500
STAGE1_PATIENCE, STAGE2_PATIENCE = 15, 50
STAGE1_LR, STAGE2_LR, WEIGHT_DECAY = 1e-3, 3e-5, 1e-5
MIN_DELTA, GRAD_CLIP = 1e-5, 5.0
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================== 2. 工具函数 ====================
def set_seed(seed=123):
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


# ==================== 3. Dataset ====================
class EnzymeDataset(Dataset):
    def __init__(self, df, esm, mol, y_mean, y_std):
        self.df, self.esm, self.mol = df.reset_index(drop=True), esm, mol
        y = np.log1p(self.df[Y_COL].to_numpy(dtype=float))
        self.y = ((y - y_mean) / y_std).astype(np.float32)

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
        return emb, mask, mol, torch.tensor(self.y[i]), str(r["idx"]), str(r["Name"]), str(r["SM"])


def collate_fn(batch):
    max_len = max(x[0].shape[0] for x in batch)
    embs, masks = [], []
    for x in batch:
        e, m = x[0], x[1]; n = min(len(e), len(m)); e, m = e[:n], m[:n]
        pad = max_len - n
        if pad: e = torch.cat([e, torch.zeros(pad, e.shape[1])]); m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])
        embs.append(e); masks.append(m)
    return torch.stack(embs), torch.stack(masks), torch.stack([x[2] for x in batch]), torch.stack([x[3] for x in batch]), [x[4] for x in batch], [x[5] for x in batch], [x[6] for x in batch]


# ==================== 4. 模型 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.esm_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        # 浓度分支已移除：SM loading 与底物一一对应（MolFormer 已覆盖），Enz 效应待实验确认后再定
        self.fc1, self.bn1 = nn.Linear(512, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2)
        self.fc4, self.dropout = nn.Linear(hidden_dim // 2, 1), nn.Dropout(0.2)

    def forward(self, esm, mask, mol):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1e-8)
        x = torch.cat([pooled, self.mol_proj(mol)], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


# ==================== 5. 加载预训练权重 ====================
def load_pretrained(model, path, device):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    old_fc1_w, old_fc1_b = ckpt.get("fc1.weight"), ckpt.get("fc1.bias")
    ckpt.pop("fc4.weight", None); ckpt.pop("fc4.bias", None)
    current = model.state_dict()
    matched = {k: v for k, v in ckpt.items() if k in current and v.shape == current[k].shape}
    ratio = sum(v.numel() for v in matched.values()) / sum(v.numel() for v in current.values())
    print(f"预训练权重匹配比例: {ratio:.4f}，匹配层数: {len(matched)}")
    if ratio < 0.80: raise RuntimeError("预训练权重匹配比例低于80%，请检查模型结构")
    model.load_state_dict(matched, strict=False)
    # fc1 输入由 768[pooled256, enz128, mol256, sm128] 变为 512[pooled256, mol256]，pooled/mol 两段按列搬运
    if old_fc1_w is not None and old_fc1_w.shape == (model.fc1.out_features, 768):
        with torch.no_grad():
            model.fc1.weight[:, 0:256] = old_fc1_w[:, 0:256]
            model.fc1.weight[:, 256:512] = old_fc1_w[:, 384:640]
            model.fc1.bias.copy_(old_fc1_b)
    nn.init.xavier_uniform_(model.fc4.weight); nn.init.zeros_(model.fc4.bias)
    return model


# ==================== 6. 回归指标 ====================
def safe_corr(func, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0: return np.nan
    try: return float(func(y, p)[0])
    except Exception: return np.nan


def regression_metrics(y, p):
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p), "Pearson": safe_corr(pearsonr, y, p), "Spearman": safe_corr(spearmanr, y, p)}


# ==================== 7. 训练与预测 ====================
def train_epoch(model, loader, optimizer, loss_fn, device, training=True):
    model.train(training); total, count = 0.0, 0
    for b in loader:
        e, mask, mol, y = [v.to(device, non_blocking=True) for v in b[:4]]
        if training and len(y) == 1: model.eval()
        pred = model(e, mask, mol); loss = loss_fn(pred, y)
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
        e, mask, mol, y = [v.to(device, non_blocking=True) for v in b[:4]]
        p = model(e, mask, mol).cpu().numpy()
        ys.extend(y.cpu().numpy()); ps.extend(p)
        meta.extend(zip(*b[4:7]))
    y_true = np.expm1(np.asarray(ys) * y_std + y_mean)
    y_pred = np.expm1(np.asarray(ps) * y_std + y_mean)
    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["True_Activity"], result["Pred_Activity"] = y_true, y_pred
    return regression_metrics(y_true, y_pred), result


def save_model(model, path):
    tmp = path + ".tmp"; torch.save(model.state_dict(), tmp); os.replace(tmp, path)


def train_stage(model, train_loader, val_loader, device, loss_fn, epochs, patience, lr, params, y_mean, y_std, save_path):
    for p in model.parameters(): p.requires_grad = False
    for p in params: p.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY)
    best, bad = np.inf, 0
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss = train_epoch(model, val_loader, None, loss_fn, device, False)
        val_metrics, _ = predict(model, val_loader, device, y_mean, y_std)
        rmse = val_metrics["RMSE"]
        print(f"Epoch {epoch:03d}/{epochs} | TrainLoss={tr_loss:.6f} | ValLoss={val_loss:.6f} | ValRMSE={rmse:.6f} | MAE={val_metrics['MAE']:.6f} | R2={val_metrics['R2']:.4f}")
        if rmse < best - MIN_DELTA:
            best, bad = rmse, 0; save_model(model, save_path); print("  >>> 保存最佳回归模型")
        else: bad += 1
        if bad >= patience: print("Early Stopping"); break
    if not os.path.isfile(save_path): raise RuntimeError("没有生成最佳模型")
    model.load_state_dict(torch.load(save_path, map_location=device))


# ==================== 8. 主程序 ====================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Task: Pure Activity Regression")
    esm, mol = load_pickle(ESM_PKL), load_pickle(MOL_PKL)
    print("ESM features:", len(esm), "| MolFormer features:", len(mol))
    df = pd.read_csv(CSV_PATH)
    required = ["idx", "Name", "SM", Y_COL]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"CSV 缺少列: {missing}")
    for c in ["idx", "Name", "SM"]: df[c] = df[c].astype(str).str.strip()
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[Y_COL]).reset_index(drop=True)
    if df["idx"].duplicated().any(): raise ValueError("idx 存在重复")
    if (df[Y_COL] <= -1).any(): raise ValueError("Activity 必须大于-1才能使用log1p")
    ids = {n: set(read_ids(n)) for n in ["train.txt", "val.txt", "test.txt"]}
    if ids["train.txt"] & ids["val.txt"] or ids["train.txt"] & ids["test.txt"] or ids["val.txt"] & ids["test.txt"]: raise ValueError("Train/Val/Test idx 存在交集")
    parts = {k: df[df.idx.isin(v)].copy().reset_index(drop=True) for k, v in [("train", ids["train.txt"]), ("val", ids["val.txt"]), ("test", ids["test.txt"])]}
    if any(x.empty for x in parts.values()): raise ValueError("Train/Val/Test 存在空数据集")
    groups = {k: set(v.Name) for k, v in parts.items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]: raise ValueError("Train/Val/Test 存在重复酶 Name")
    for k, d in parts.items():
        if set(d.Name) - set(esm) or set(d.SM) - set(mol): raise KeyError(f"{k} 缺少 ESM 或 MolFormer 特征")
    y_log = np.log1p(parts["train"][Y_COL])
    y_mean, y_std = float(y_log.mean()), float(y_log.std())
    if not np.isfinite(y_std) or y_std < 1e-12: y_std = 1.0
    with open(os.path.join(MODEL_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"activity_mean": y_mean, "activity_std": y_std, "transform": "log1p+zscore"}, f, ensure_ascii=False, indent=2)
    print("Samples:", {k: len(v) for k, v in parts.items()}, "| Enzymes:", {k: len(v) for k, v in groups.items()})
    for k, d in parts.items(): print(f"{k}: zero={(d[Y_COL] == 0).sum()}, mean={d[Y_COL].mean():.4f}, std={d[Y_COL].std():.4f}, max={d[Y_COL].max():.4f}")
    datasets = {k: EnzymeDataset(v, esm, mol, y_mean, y_std) for k, v in parts.items()}
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda", collate_fn=collate_fn)
    loaders = {"train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **kw), "val": DataLoader(datasets["val"], shuffle=False, **kw), "test": DataLoader(datasets["test"], shuffle=False, **kw)}
    model = load_pretrained(EnzymeSubstrateRegressor().to(device), PRETRAINED_MODEL, device)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    print("\n========== STAGE 1: TRAIN FC4 ==========")
    stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")
    train_stage(model, loaders["train"], loaders["val"], device, loss_fn, STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR, list(model.fc4.parameters()), y_mean, y_std, stage1_path)
    print("\n========== STAGE 2: FINE-TUNE FULL MODEL ==========")
    best_path = os.path.join(MODEL_DIR, "best_regression.pth")
    train_stage(model, loaders["train"], loaders["val"], device, loss_fn, STAGE2_EPOCHS, STAGE2_PATIENCE, STAGE2_LR, list(model.parameters()), y_mean, y_std, best_path)
    print("\n========== TEST: PURE REGRESSION ==========")
    test_metrics, test_result = predict(model, loaders["test"], device, y_mean, y_std)
    for k, v in test_metrics.items(): print(f"{k}: {v:.6f}")
    test_result.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False, encoding="utf-8-sig")
    val_metrics, val_result = predict(model, loaders["val"], device, y_mean, y_std)
    val_result.to_csv(os.path.join(MODEL_DIR, "val_predictions.csv"), index=False, encoding="utf-8-sig")
    print("Best model:", best_path)
    print("Scaler:", os.path.join(MODEL_DIR, "scaler.json"))
    print("Output:", MODEL_DIR)


if __name__ == "__main__": main()
