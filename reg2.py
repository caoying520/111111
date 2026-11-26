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
PRETRAINED_CSV = "/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/data/酶筛/merged_KRD_S2_KRD_M_add.csv"

CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/13k_std.csv"
ESM_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/esm_144KRD.pkl"
MOL_PKL = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/SM96_std.pkl"
SPLIT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/split_Activity_enz_clean"
MODEL_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/KRD/13K/model_Activity_reg_enz/split2/split_去重复酶权重_加浓度"

Y_COL = "Activity"

# 当前 13k_std.csv 如果已经有标准化浓度列，使用下面设置。
# 如果 CSV 只有原始 g/L 列，将 CONC_MODE 改为 "raw_gl"，并按实际列名修改 SM_COL、ENZ_COL。
CONC_MODE = "normalized"       # "normalized" 或 "raw_gl"
SM_COL, ENZ_COL = "SM_norm", "Enz_norm"
# raw_gl 模式默认列名：
# SM_COL, ENZ_COL = "SM loading (g/L)", "Enzyme loading (g/L)"

# 预训练数据中出现过的酶序列降权；只影响训练集，不影响验证/测试评价。
USE_SAMPLE_WEIGHT = True
OVERLAP_WEIGHT = 0.3

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


def clean_sequence(x):
    if pd.isna(x):
        return ""
    return "".join(str(x).upper().split())


def safe_std(x):
    s = float(np.std(x))
    return s if np.isfinite(s) and s > 1e-12 else 1.0


# ==================== 3. Dataset ====================
class EnzymeDataset(Dataset):
    def __init__(self, df, esm, mol, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.esm, self.mol = esm, mol

        y = np.log1p(self.df[Y_COL].to_numpy(dtype=float))
        self.y = ((y - y_mean) / y_std).astype(np.float32)

        # normalized 模式直接使用 CSV 中已经标准化的浓度；
        # raw_gl 模式在 main() 中已完成 log1p + train-set z-score。
        self.sm = self.df["_sm_input"].to_numpy(dtype=np.float32)
        self.enz = self.df["_enz_input"].to_numpy(dtype=np.float32)

        # 没有 sample_w 的验证/测试样本默认权重为 1。
        self.w = self.df["sample_w"].to_numpy(dtype=np.float32) if "sample_w" in self.df.columns else np.ones(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        name, sm_id = str(r["Name"]), str(r["SM"])

        if name not in self.esm:
            raise KeyError(f"ESM 特征中不存在酶：{name}")
        if sm_id not in self.mol:
            raise KeyError(f"MolFormer 特征中不存在底物：{sm_id}")

        e = self.esm[name]
        emb = torch.as_tensor(np.asarray(e["emb"]), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(e["mask"]), dtype=torch.bool).reshape(-1)
        n = min(len(emb), len(mask))
        emb, mask = emb[:n], mask[:n]
        if n == 0:
            raise ValueError(f"{name} 的 ESM 序列长度为0")

        mol = torch.as_tensor(np.asarray(self.mol[sm_id]), dtype=torch.float32).reshape(-1)
        if emb.ndim != 2 or emb.shape[1] != 1280 or mol.numel() != 768:
            raise ValueError(f"ESM/MolFormer 特征维度错误：{name}, {sm_id}")

        return (
            emb, mask, mol,
            torch.tensor(self.sm[i], dtype=torch.float32),
            torch.tensor(self.enz[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            str(r["idx"]), name, sm_id,
            torch.tensor(self.w[i], dtype=torch.float32)
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
        torch.stack(embs),
        torch.stack(masks),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        torch.stack([x[5] for x in batch]),
        [x[6] for x in batch],
        [x[7] for x in batch],
        [x[8] for x in batch],
        torch.stack([x[9] for x in batch])
    )


# ==================== 4. 模型：ESM + MolFormer + 两个浓度分支 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1, hidden_dim=256):
        super().__init__()

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 256)
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 256)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.esm_encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )

        # 每个浓度标量映射到128维；模块名称需与预训练 checkpoint 对应。
        self.sm_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128)
        )
        self.enz_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128)
        )

        # 拼接顺序与原始预训练模型一致：[ESM256, Enz128, Mol256, SM128]。
        self.fc1, self.bn1 = nn.Linear(768, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2)
        self.fc4, self.dropout = nn.Linear(hidden_dim // 2, 1), nn.Dropout(0.2)

    def forward(self, esm, mask, mol, sm, enz):
        x = self.esm_proj(esm)
        x = self.esm_encoder(x, src_key_padding_mask=~mask)

        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1e-8)

        enz_feat = self.enz_proj(enz.unsqueeze(-1))
        mol_feat = self.mol_proj(mol)
        sm_feat = self.sm_proj(sm.unsqueeze(-1))

        x = torch.cat([pooled, enz_feat, mol_feat, sm_feat], dim=-1)
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

    # 只重新初始化最终 Activity 回归头。
    ckpt.pop("fc4.weight", None)
    ckpt.pop("fc4.bias", None)

    current = model.state_dict()
    matched = {
        k: v for k, v in ckpt.items()
        if k in current and torch.is_tensor(v) and v.shape == current[k].shape
    }

    total_numel = sum(v.numel() for v in current.values())
    matched_numel = sum(v.numel() for v in matched.values())
    ratio = matched_numel / max(total_numel, 1)

    print(f"预训练权重匹配比例（不含新 fc4）：{ratio:.4f}")
    print(f"匹配参数张量数：{len(matched)} / {len(current)}")

    missing = [k for k in current if k not in matched]
    print("未匹配参数：", missing)

    # 关键：浓度分支必须确实从预训练模型加载，不能仅凭总匹配率判断。
    sm_loaded = any(k.startswith("sm_proj.") for k in matched)
    enz_loaded = any(k.startswith("enz_proj.") for k in matched)
    fc1_loaded = "fc1.weight" in matched and matched["fc1.weight"].shape == current["fc1.weight"].shape

    if ratio < 0.80:
        raise RuntimeError("预训练权重匹配比例低于80%，请检查模型结构。")
    if not fc1_loaded:
        raise RuntimeError("fc1.weight 未能完整匹配；当前768维拼接结构与预训练模型不一致。")
    if not sm_loaded or not enz_loaded:
        raise RuntimeError(
            "至少一个浓度分支没有从预训练权重加载。"
            "请检查 checkpoint 中浓度分支的真实参数名称/维度，"
            "不能把随机初始化的浓度分支误认为已继承预训练知识。"
        )

    model.load_state_dict(matched, strict=False)
    nn.init.xavier_uniform_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)
    return model


# ==================== 6. 回归指标 ====================
def safe_corr(func, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    try:
        return float(func(y, p)[0])
    except Exception:
        return np.nan


def regression_metrics(y, p):
    return {
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "MAE": mean_absolute_error(y, p),
        "R2": r2_score(y, p),
        "Pearson": safe_corr(pearsonr, y, p),
        "Spearman": safe_corr(spearmanr, y, p)
    }


# ==================== 7. 训练与预测 ====================
def train_epoch(model, loader, optimizer, loss_fn, device, training=True):
    model.train(training)
    total, total_weight = 0.0, 0.0

    for b in loader:
        e, mask, mol, sm, enz, y = [
            v.to(device, non_blocking=True) for v in b[:6]
        ]
        w = b[9].to(device, non_blocking=True)

        # BatchNorm 在 batch_size=1 时无法正常训练；该 batch 暂时使用 eval 模式。
        if training and len(y) == 1:
            model.eval()

        pred = model(e, mask, mol, sm, enz)
        loss_vec = loss_fn(pred, y)

        if training:
            # 仅训练集使用重叠酶权重；验证集损失保持未加权。
            loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-8)
            denom = float(w.sum().detach().cpu())
        else:
            loss = loss_vec.mean()
            denom = float(len(y))

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if len(y) == 1:
                model.train()

        total += loss.item() * denom
        total_weight += denom

    return total / max(total_weight, 1e-8)


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
        meta.extend(zip(*b[6:9]))

    y_true = np.expm1(np.asarray(ys) * y_std + y_mean)
    y_pred = np.expm1(np.asarray(ps) * y_std + y_mean)

    result = pd.DataFrame(meta, columns=["idx", "Name", "SM"])
    result["True_Activity"] = y_true
    result["Pred_Activity"] = y_pred
    return regression_metrics(y_true, y_pred), result


def save_model(model, path):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def train_stage(model, train_loader, val_loader, device, loss_fn,
                epochs, patience, lr, params, y_mean, y_std, save_path):
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=WEIGHT_DECAY
    )

    best, bad = np.inf, 0
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss = train_epoch(model, val_loader, None, loss_fn, device, False)
        val_metrics, _ = predict(model, val_loader, device, y_mean, y_std)
        rmse = val_metrics["RMSE"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"TrainLoss(weighted)={tr_loss:.6f} | ValLoss={val_loss:.6f} | "
            f"ValRMSE={rmse:.6f} | MAE={val_metrics['MAE']:.6f} | "
            f"R2={val_metrics['R2']:.4f}"
        )

        if rmse < best - MIN_DELTA:
            best, bad = rmse, 0
            save_model(model, save_path)
            print("  >>> 保存最佳回归模型")
        else:
            bad += 1

        if bad >= patience:
            print("Early Stopping")
            break

    if not os.path.isfile(save_path):
        raise RuntimeError("没有生成最佳模型")
    model.load_state_dict(torch.load(save_path, map_location=device))


# ==================== 8. 主程序 ====================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Task: Pure Activity Regression")
    print("Concentration mode:", CONC_MODE)
    print("Overlap sample weighting:", USE_SAMPLE_WEIGHT, "| overlap weight:", OVERLAP_WEIGHT)

    esm, mol = load_pickle(ESM_PKL), load_pickle(MOL_PKL)
    print("ESM features:", len(esm), "| MolFormer features:", len(mol))

    df = pd.read_csv(CSV_PATH)
    required = ["idx", "Name", "SM", Y_COL, "sequence", SM_COL, ENZ_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"CSV 缺少列: {missing}。"
            f"请核对 SM_COL/ENZ_COL；如果是原始 g/L 列，设置 CONC_MODE='raw_gl'。"
        )

    for c in ["idx", "Name", "SM"]:
        df[c] = df[c].astype(str).str.strip()

    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df[SM_COL] = pd.to_numeric(df[SM_COL], errors="coerce")
    df[ENZ_COL] = pd.to_numeric(df[ENZ_COL], errors="coerce")
    df = df.dropna(subset=[Y_COL, SM_COL, ENZ_COL]).reset_index(drop=True)

    if df["idx"].duplicated().any():
        raise ValueError("idx 存在重复")
    if (df[Y_COL] <= -1).any():
        raise ValueError("Activity 必须大于-1才能使用log1p")
    if (df[SM_COL] < 0).any() or (df[ENZ_COL] < 0).any():
        raise ValueError("浓度不能为负数")

    ids = {n: set(read_ids(n)) for n in ["train.txt", "val.txt", "test.txt"]}
    if (ids["train.txt"] & ids["val.txt"] or
        ids["train.txt"] & ids["test.txt"] or
        ids["val.txt"] & ids["test.txt"]):
        raise ValueError("Train/Val/Test idx 存在交集")

    parts = {
        k: df[df["idx"].isin(v)].copy().reset_index(drop=True)
        for k, v in [
            ("train", ids["train.txt"]),
            ("val", ids["val.txt"]),
            ("test", ids["test.txt"])
        ]
    }

    if any(x.empty for x in parts.values()):
        raise ValueError("Train/Val/Test 存在空数据集")

    groups = {k: set(v["Name"]) for k, v in parts.items()}
    if (groups["train"] & groups["val"] or
        groups["train"] & groups["test"] or
        groups["val"] & groups["test"]):
        raise ValueError("Train/Val/Test 存在重复酶 Name")

    for k, d in parts.items():
        if set(d["Name"]) - set(esm):
            raise KeyError(f"{k} 缺少 ESM 特征")
        if set(d["SM"]) - set(mol):
            raise KeyError(f"{k} 缺少 MolFormer 特征")

    # Activity：只用训练集统计量做 log1p + z-score。
    y_log = np.log1p(parts["train"][Y_COL].to_numpy(dtype=float))
    y_mean, y_std = float(y_log.mean()), safe_std(y_log)

    # 浓度输入：normalized 模式直接使用已有标准化值；
    # raw_gl 模式先 log1p，再仅用训练集均值/标准差标准化。
    if CONC_MODE == "normalized":
        for k, d in parts.items():
            d["_sm_input"] = d[SM_COL].astype(float)
            d["_enz_input"] = d[ENZ_COL].astype(float)
        sm_mean = sm_std = enz_mean = enz_std = None

    elif CONC_MODE == "raw_gl":
        sm_train_log = np.log1p(parts["train"][SM_COL].to_numpy(dtype=float))
        enz_train_log = np.log1p(parts["train"][ENZ_COL].to_numpy(dtype=float))
        sm_mean, sm_std = float(sm_train_log.mean()), safe_std(sm_train_log)
        enz_mean, enz_std = float(enz_train_log.mean()), safe_std(enz_train_log)

        for k, d in parts.items():
            sm_log = np.log1p(d[SM_COL].to_numpy(dtype=float))
            enz_log = np.log1p(d[ENZ_COL].to_numpy(dtype=float))
            d["_sm_input"] = ((sm_log - sm_mean) / sm_std).astype(np.float32)
            d["_enz_input"] = ((enz_log - enz_mean) / enz_std).astype(np.float32)
    else:
        raise ValueError("CONC_MODE 只能是 'normalized' 或 'raw_gl'")

    # ---------- 预训练数据重叠酶降权 ----------
    # 这里按 sequence 是否出现在 PRETRAINED_CSV 中判断；
    # 不是按当前 Activity 数据中某个 Name 出现了多少次来降权。
    for k, d in parts.items():
        d["sample_w"] = np.ones(len(d), dtype=np.float32)

    if USE_SAMPLE_WEIGHT:
        if not os.path.isfile(PRETRAINED_CSV):
            raise FileNotFoundError(PRETRAINED_CSV)

        pre_df = pd.read_csv(PRETRAINED_CSV, low_memory=False)
        if "sequence" not in pre_df.columns:
            raise KeyError("预训练 CSV 缺少 sequence 列，无法判定预训练重叠酶")

        pre_seqs = set(pre_df["sequence"].map(clean_sequence))
        pre_seqs.discard("")

        for k, d in parts.items():
            if "sequence" not in d.columns:
                raise KeyError("Activity CSV 缺少 sequence 列，无法判定预训练重叠酶")

            overlap = d["sequence"].map(clean_sequence).isin(pre_seqs)
            if k == "train":
                d["sample_w"] = np.where(
                    overlap, OVERLAP_WEIGHT, 1.0
                ).astype(np.float32)

            print(
                f"{k}: 预训练序列重叠行 {int(overlap.sum())}/{len(d)} "
                f"({overlap.mean() * 100:.1f}%)；"
                f"{'训练权重=' + str(OVERLAP_WEIGHT) if k == 'train' else '评价不降权'}"
            )

    # 保存 scaler 和训练配置，便于推理阶段复用。
    scaler = {
        "activity_mean": y_mean,
        "activity_std": y_std,
        "activity_transform": "log1p+train_zscore",
        "concentration_mode": CONC_MODE,
        "SM_COL": SM_COL,
        "ENZ_COL": ENZ_COL,
        "SM_mean": sm_mean,
        "SM_std": sm_std,
        "Enz_mean": enz_mean,
        "Enz_std": enz_std,
        "overlap_weighting": USE_SAMPLE_WEIGHT,
        "overlap_weight": OVERLAP_WEIGHT
    }
    with open(os.path.join(MODEL_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    print("Samples:", {k: len(v) for k, v in parts.items()},
          "| Enzymes:", {k: len(v) for k, v in groups.items()})
    for k, d in parts.items():
        print(
            f"{k}: zero={(d[Y_COL] == 0).sum()}, "
            f"mean={d[Y_COL].mean():.4f}, std={d[Y_COL].std():.4f}, "
            f"max={d[Y_COL].max():.4f}, "
            f"mean_sample_w={d['sample_w'].mean():.4f}"
        )

    datasets = {
        k: EnzymeDataset(v, esm, mol, y_mean, y_std)
        for k, v in parts.items()
    }

    kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn
    )
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **kw),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **kw),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **kw)
    }

    model = load_pretrained(
        EnzymeSubstrateRegressor().to(device),
        PRETRAINED_MODEL, device
    )

    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")

    print("\n========== STAGE 1: TRAIN FC4 ==========")
    stage1_path = os.path.join(MODEL_DIR, "stage1_best.pth")
    train_stage(
        model, loaders["train"], loaders["val"], device, loss_fn,
        STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR,
        list(model.fc4.parameters()), y_mean, y_std, stage1_path
    )

    print("\n========== STAGE 2: FINE-TUNE FULL MODEL ==========")
    best_path = os.path.join(MODEL_DIR, "best_regression.pth")
    train_stage(
        model, loaders["train"], loaders["val"], device, loss_fn,
        STAGE2_EPOCHS, STAGE2_PATIENCE, STAGE2_LR,
        list(model.parameters()), y_mean, y_std, best_path
    )

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

    val_metrics, val_result = predict(
        model, loaders["val"], device, y_mean, y_std
    )
    print("Best validation metrics:", val_metrics)
    val_result.to_csv(
        os.path.join(MODEL_DIR, "val_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("Best model:", best_path)
    print("Scaler:", os.path.join(MODEL_DIR, "scaler.json"))
    print("Output:", MODEL_DIR)


if __name__ == "__main__":
    main()
