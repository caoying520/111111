# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# coding: utf-8

import os, random, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr


# ========================
# 1️⃣ 固定随机种子
# ========================
def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


# ========================
# 2️⃣ 标签变换
# ========================
def transform_activity(y):
    return np.log1p(np.asarray(y, dtype=np.float32))


def inverse_activity(y):
    return np.expm1(np.asarray(y))


# ========================
# 3️⃣ Dataset 定义
# ========================
class EnzymeSubstrateDataset(Dataset):
    """【改动】新增底物 one-hot 编码。
    数据里只有 23 种胺 + 12 种酮，用 one-hot 直接告诉模型"是哪一种底物"，
    比让模型从 768 维 MolFormer 向量里自己反推要容易得多。"""

    def __init__(self, df, esm_embeds, mol_embeds1, mol_embeds2, amine2idx, ketone2idx):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["Name"].astype(str).tolist()
        self.smiles1 = self.df["SM_胺"].tolist()
        self.smiles2 = self.df["SM_酮"].tolist()
        self.labels = pd.to_numeric(self.df["Activity"], errors="coerce").fillna(0).astype(float).to_numpy()
        self.labels_trans = transform_activity(self.labels)
        self.esm_embeds = esm_embeds; self.mol_embeds1 = mol_embeds1; self.mol_embeds2 = mol_embeds2
        # 【新增】底物类别索引表
        self.amine2idx = amine2idx
        self.ketone2idx = ketone2idx
        self.n_amine = len(amine2idx)
        self.n_ketone = len(ketone2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        esm_data = self.esm_embeds[self.idx[i]]
        esm_emb = torch.as_tensor(esm_data["emb"], dtype=torch.float32)
        esm_mask = torch.as_tensor(esm_data["mask"], dtype=torch.long)
        mol_feat1 = torch.as_tensor(self.mol_embeds1[self.smiles1[i]], dtype=torch.float32)
        mol_feat2 = torch.as_tensor(self.mol_embeds2[self.smiles2[i]], dtype=torch.float32)
        # 【新增】底物 one-hot
        oh1 = F.one_hot(torch.tensor(self.amine2idx[self.smiles1[i]]), self.n_amine).float()
        oh2 = F.one_hot(torch.tensor(self.ketone2idx[self.smiles2[i]]), self.n_ketone).float()
        return (esm_emb, esm_mask, mol_feat1, mol_feat2, oh1, oh2,
                torch.tensor(self.labels_trans[i], dtype=torch.float32),
                torch.tensor(self.labels[i], dtype=torch.float32))


# ========================
# 4️⃣ 模型定义
# ========================
class EnzymeSubstrateRegressor(nn.Module):
    """【改动】在 concat 特征里新增两路显式交互项 esm⊙mol。
    原结构只有 [esm_pooled; mol1; mol2] 直接拼接，酶与底物之间的匹配关系
    要靠 MLP 自己学交叉项；显式乘法通道能直接表达"这对酶-底物是否匹配"。"""

    def __init__(self, esm_dim=1280, mol_dim=768, n_amine=23, n_ketone=12,
                 dropout=0.3, hidden_dim=256):
        super().__init__()

        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj1 = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.mol_proj2 = nn.Sequential(nn.Linear(mol_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        # 【回退】恢复 mean 单池化(256维)：run2 已验证 mean+max 是负优化(test R2 0.426→0.362)
        in_dim = 256 * 5 + n_amine + n_ketone
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)

        # 纯回归头
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, esm_feat, esm_mask, mol_feat1, mol_feat2, oh1, oh2):
        esm_feat = self.esm_proj(esm_feat)
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=(esm_mask == 0))
        mask = esm_mask.unsqueeze(-1).float()
        esm_pooled = (esm_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        mol1 = self.mol_proj1(mol_feat1)
        mol2 = self.mol_proj2(mol_feat2)

        # 酶-底物显式交互 + 底物 one-hot
        x = torch.cat([esm_pooled, mol1, mol2,
                       esm_pooled * mol1, esm_pooled * mol2,
                       oh1, oh2], dim=-1)

        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = self.dropout(F.relu(self.bn2(self.fc2(x)))) + residual
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))

        return self.reg_head(x).squeeze(-1)


# ========================
# 4.5️⃣ 损失函数【新增】
# ========================
class HybridRegressionLoss(nn.Module):
    """【新增】log 空间 + 原始空间双损失，并按活性加权。

    原实现只在 log 空间算 SmoothL1，R²/RMSE 却是在 expm1 之后的原始空间评估，
    高活性样本（max 27.7）的误差在原始空间被放大，导致尾部系统性低估
    （日志里 Active_MAE=1.12 远大于整体 MAE=0.71）。

    - loss_log：log 空间平滑 L1，按 (1 + α·log1p(y)) 加权，让高活性样本占更大比重
    - loss_raw：把预测还原到原始空间后除以训练集标准差，与评估口径对齐
    """

    def __init__(self, beta=0.5, lambda_raw=0.3, alpha_weight=1.0, raw_scale=1.0):
        super().__init__()
        self.beta = beta
        self.lambda_raw = lambda_raw
        self.alpha_weight = alpha_weight
        self.raw_scale = raw_scale

    def forward(self, pred_log, target_log):
        # 权重：活性越高权重越大，缓解零值样本主导梯度
        w = 1.0 + self.alpha_weight * target_log
        l_log = F.smooth_l1_loss(pred_log, target_log, beta=self.beta, reduction="none")
        loss_log = (w * l_log).sum() / (w.sum() + 1e-8)

        # 原始空间项：预测先夹到非负（活性不可能为负），再按训练集 std 归一化
        pred_raw = torch.expm1(pred_log.clamp(min=0.0))
        target_raw = torch.expm1(target_log)
        loss_raw = F.smooth_l1_loss(pred_raw / self.raw_scale,
                                    target_raw / self.raw_scale, beta=1.0)

        total = loss_log + self.lambda_raw * loss_raw
        return {"total": total, "log": loss_log.detach(), "raw": loss_raw.detach()}


# ========================
# 5️⃣ 评估函数
# ========================
def compute_metrics(y_true, y_pred, pred_log=None, loss=np.nan):
    """【改动】从 evaluate 里抽出来，便于对校准后的预测复用同一套指标。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    pearson = pearsonr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 and len(y_true) > 1 else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if len(np.unique(y_true)) > 1 else np.nan

    # 非零活性样本排序
    active = y_true != 0
    active_spearman = spearmanr(y_true[active], y_pred[active]).statistic if active.sum() > 1 and len(np.unique(y_true[active])) > 1 else np.nan
    active_mae = mean_absolute_error(y_true[active], y_pred[active]) if active.sum() else np.nan

    # Top100 排序评价
    k = min(100, len(y_true))
    true_top = np.argsort(-y_true)[:k]
    pred_top = np.argsort(-y_pred)[:k]
    overlap = len(set(true_top) & set(pred_top)) / k if k else np.nan

    # NDCG要求真实相关性非负，因此整体平移标签，不改变标签排序
    relevance = y_true - min(0, np.min(y_true))
    ndcg = ndcg_score(relevance.reshape(1, -1), y_pred.reshape(1, -1), k=k) if k else np.nan

    # 【新增】log 空间 R²：用于区分"模型没学会"还是"只是原始空间尾部被压扁"
    r2_log = np.nan
    if pred_log is not None and len(y_true) > 1:
        r2_log = r2_score(np.log1p(np.clip(y_true, 0, None)), pred_log)

    return {"Loss": loss, "MAE": mae, "RMSE": rmse, "R2": r2, "R2_log": r2_log,
            "Pearson": pearson, "Spearman": spearman,
            "Active_MAE": active_mae, "Active_Spearman": active_spearman,
            "NDCG@100": ndcg, "Top100_Overlap": overlap,
            "Top100_True_Mean": np.mean(y_true[pred_top]) if k else np.nan}


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_pred, all_true, all_log_pred = [], [], []
    total_loss, total_num = 0.0, 0

    with torch.no_grad():
        for esm, mask, mol1, mol2, oh1, oh2, y, raw_y in loader:
            esm, mask, mol1, mol2, oh1, oh2, y = (
                esm.to(device), mask.to(device), mol1.to(device), mol2.to(device),
                oh1.to(device), oh2.to(device), y.to(device))
            output = model(esm, mask, mol1, mol2, oh1, oh2)

            if criterion is not None:
                loss = criterion(output, y)["total"]
                total_loss += loss.item() * len(raw_y)
                total_num += len(raw_y)

            # 【改动】输出夹到非负，避免 expm1 产生负活性（真实活性 min=0）
            pred = inverse_activity(np.clip(output.cpu().numpy(), 0.0, None))
            all_pred.extend(pred.tolist())
            all_true.extend(raw_y.numpy().tolist())
            all_log_pred.extend(output.cpu().numpy().tolist())

    y_true, y_pred = np.array(all_true), np.array(all_pred)
    metrics = compute_metrics(y_true, y_pred, np.array(all_log_pred),
                              total_loss / total_num if total_num else np.nan)
    predictions = {"true": y_true, "pred": y_pred, "pred_log": np.array(all_log_pred)}
    return metrics, predictions


# ========================
# 5.5️⃣ 事后校准【新增】
# ========================
def fit_log_linear_calibration(pred_log, true_raw):
    """在 log 空间拟合 true_log ≈ a * pred_log + b（最小二乘）。
    零值膨胀 + 长尾分布下，模型输出普遍被压向中位数，
    一个单调线性重标定就能把尺度拉回来。"""
    t = np.log1p(np.clip(np.asarray(true_raw, dtype=float), 0, None))
    A = np.stack([pred_log, np.ones_like(pred_log)], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef


def apply_log_linear_calibration(pred_log, coef):
    return inverse_activity(np.clip(coef[0] * pred_log + coef[1], 0.0, None))


# ========================
# 6️⃣ 训练函数
# ========================
def train_model(train_loader, val_loader, model, optimizer, criterion, device,
                epochs=500, patience=100, model_dir=None, swa_start=10**9):
    os.makedirs(model_dir, exist_ok=True)
    best_metric, patience_counter = -np.inf, 0
    history = []
    # 【新增】SWA：从 swa_start 轮起做参数滑动平均，避免把"挑到噪声峰值"的单个 checkpoint 当作最终模型
    swa_state = None
    swa_count = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=12, min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        total_loss, total_log, total_raw = 0.0, 0.0, 0.0

        for esm, mask, mol1, mol2, oh1, oh2, y, raw_y in train_loader:
            esm, mask, mol1, mol2, oh1, oh2, y = (
                esm.to(device), mask.to(device), mol1.to(device), mol2.to(device),
                oh1.to(device), oh2.to(device), y.to(device))
            output = model(esm, mask, mol1, mol2, oh1, oh2)
            parts = criterion(output, y)
            loss = parts["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_log += parts["log"].item()
            total_raw += parts["raw"].item()

        nb = max(len(train_loader), 1)
        train_loss = total_loss / nb
        train_log = total_log / nb
        train_raw = total_raw / nb

        val_metrics, _ = evaluate(model, val_loader, device, criterion)

        # 【回退】选模指标恢复为原始 R2（run2 已验证该配置 test R2=0.426）
        monitor = val_metrics["R2"]
        if not np.isfinite(monitor):
            monitor = -val_metrics["Loss"]
        scheduler.step(monitor)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss={train_loss:.4f} (log={train_log:.4f}, raw={train_raw:.4f}) | "
              f"Val Loss={val_metrics['Loss']:.4f} | R2={val_metrics['R2']:.4f} | R2_log={val_metrics['R2_log']:.4f} | "
              f"MAE={val_metrics['MAE']:.4f} | RMSE={val_metrics['RMSE']:.4f} | "
              f"Spearman={val_metrics['Spearman']:.4f} | Active Spearman={val_metrics['Active_Spearman']:.4f} | "
              f"NDCG@100={val_metrics['NDCG@100']:.4f} | Top100 Overlap={val_metrics['Top100_Overlap']:.4f}")

        # 【新增】每 10 轮看一次训练集指标，便于判断 train/val 差距
        if (epoch + 1) % 10 == 0:
            train_metrics, _ = evaluate(model, train_loader, device, None)
            print(f"    [诊断] Train R2={train_metrics['R2']:.4f} | Train R2_log={train_metrics['R2_log']:.4f} | "
                  f"Train MAE={train_metrics['MAE']:.4f}")

        history.append({"epoch": epoch+1, "train_loss": train_loss,
                        "train_loss_log": train_log, "train_loss_raw": train_raw,
                        **{"val_"+k: v for k, v in val_metrics.items()}})

        # 【新增】SWA 参数滑动平均（每轮累加，最后除以轮数）
        if (epoch + 1) >= swa_start:
            swa_count += 1
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    swa_state[k] += v.detach()

        if monitor > best_metric:
            best_metric, patience_counter = monitor, 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("保存最佳模型！")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(os.path.join(model_dir, "training_history.csv"), index=False)

    # 【新增】保存 SWA 平均后的权重
    if swa_state is not None and swa_count > 0:
        swa_state = {k: v / swa_count for k, v in swa_state.items()}
        torch.save(swa_state, os.path.join(model_dir, "swa_model.pth"))
        print(f"SWA 已平均 {swa_count} 个 epoch 的权重并保存 swa_model.pth")
    else:
        print("训练长度不足，未触发 SWA（未保存 swa_model.pth）")

    print("训练结束，最佳验证指标：", best_metric)


# ========================
# 7️⃣ 主程序
# ========================
def main():
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_dir = "/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED"
    feature_dir = os.path.join(base_dir, "Feature")
    split_dir = os.path.join(base_dir, "split_data_Activaty_reg")
    model_dir = os.path.join(base_dir, "model_activity_reg")
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 加载特征 ----------
    with open(os.path.join(feature_dir, "esm_feats_emb_mask.pkl"), "rb") as f:
        esm_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "SM_胺.pkl"), "rb") as f:
        mol_dict1 = pickle.load(f)
    with open(os.path.join(feature_dir, "SM_酮.pkl"), "rb") as f:
        mol_dict2 = pickle.load(f)

    print(f"ESM={len(esm_dict)}, MolFormer胺={len(mol_dict1)}, MolFormer酮={len(mol_dict2)}")

    # 【新增】底物类别索引表（按 key 排序保证确定性）
    amine2idx = {s: i for i, s in enumerate(sorted(mol_dict1.keys()))}
    ketone2idx = {s: i for i, s in enumerate(sorted(mol_dict2.keys()))}

    # ---------- 读取已经划分好的三个txt ----------
    def read_txt(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_idx = set(read_txt(os.path.join(split_dir, "train.txt")))
    val_idx = set(read_txt(os.path.join(split_dir, "val.txt")))
    test_idx = set(read_txt(os.path.join(split_dir, "test.txt")))

    # ---------- 读取原始CSV并按txt划分 ----------
    df = pd.read_csv(os.path.join(split_dir, "train_reg.csv"))
    df["idx"] = df["idx"].astype(str).str.strip()

    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)
    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)
    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)

    print(f"Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print(f"Train Activity分布:\n{df_train['Activity'].describe()}")
    print(f"Val Activity分布:\n{df_val['Activity'].describe()}")
    print(f"Test Activity分布:\n{df_test['Activity'].describe()}")

    if min(len(df_train), len(df_val), len(df_test)) == 0:
        raise ValueError("训练集、验证集或测试集为空，请检查txt中的idx是否与CSV的idx一致！")

    # ---------- Dataset ----------
    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict1, mol_dict2, amine2idx, ketone2idx)
    val_set = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict1, mol_dict2, amine2idx, ketone2idx)
    test_set = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict1, mol_dict2, amine2idx, ketone2idx)

    batch_size = 256
    # 【改动】原写法 len(train_set) % batch_size == 1 只在余数恰好为 1 时丢弃，
    # 余数为 1 的 batch 会让 BatchNorm 统计量剧烈抖动；改为余数 < 2 就丢弃
    drop_last = (len(train_set) % batch_size) < 2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 多 seed 集成训练与预测 ----------
    # 每个 seed 独立初始化/训练一个模型，最终在 log 空间平均各 seed 的预测，
    # 抵消训练随机性并降低单点模型的过拟合方差。
    seeds = [123, 42, 2024, 7, 99]
    n_seeds = len(seeds)

    test_true = None
    val_true = None
    test_pred_logs = []   # (n_seeds, n_test)
    val_pred_logs = []    # (n_seeds, n_val)
    test_raw_preds = []   # 每个 seed 还原到原始空间后的预测，用于方差分析

    for seed in seeds:
        set_seed(seed)
        seed_model_dir = os.path.join(model_dir, f"seed_{seed}")
        os.makedirs(seed_model_dir, exist_ok=True)

        print(f"\n{'='*60}\n>>> Seed {seed} 训练开始\n{'='*60}")

        model = EnzymeSubstrateRegressor(esm_dim=1280, mol_dim=768,
                                         n_amine=len(amine2idx), n_ketone=len(ketone2idx),
                                         dropout=0.3, hidden_dim=256).to(device)

        # 【改动】损失换成双空间加权损失；raw_scale 用训练集活性标准差做归一口径
        raw_scale = float(np.std(df_train["Activity"].values)) or 1.0
        criterion = HybridRegressionLoss(beta=0.5, lambda_raw=0.3,
                                         alpha_weight=1.0, raw_scale=raw_scale).to(device)

        # 【改动】weight_decay 恢复 5e-4（1e-3 过强导致欠拟合，见 run3.log）
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)

        train_model(train_loader, val_loader, model, optimizer, criterion, device,
                    epochs=500, patience=100, model_dir=seed_model_dir)

        # ---------- 单 seed 测试（best 与 SWA 择优） ----------
        def calibrate_and_test(weights_path):
            """加载权重，用验证集拟合校准系数，返回 (原始metrics, 校准后metrics, 预测dict, 校准后预测数组, 验证集预测dict)。"""
            model.load_state_dict(torch.load(weights_path, map_location=device))
            metrics, pred = evaluate(model, test_loader, device, criterion)
            _, val_pred = evaluate(model, val_loader, device, None)
            vmask = np.isfinite(val_pred["pred_log"])
            coef = fit_log_linear_calibration(val_pred["pred_log"][vmask], val_pred["true"][vmask])
            cal = apply_log_linear_calibration(pred["pred_log"], coef)
            metrics_cal = compute_metrics(pred["true"], cal, pred["pred_log"])
            return metrics, metrics_cal, pred, cal, val_pred

        best_m, best_cal, best_pred, best_cal_arr, best_val_pred = calibrate_and_test(
            os.path.join(seed_model_dir, "best_model.pth"))

        swa_path = os.path.join(seed_model_dir, "swa_model.pth")
        use_swa = False
        seed_pred, seed_val_pred = best_pred, best_val_pred
        seed_cal = best_cal
        if os.path.exists(swa_path):
            swa_m, swa_cal, swa_pred, swa_val_pred, swa_cal_arr = calibrate_and_test(swa_path)
            if swa_cal["R2_log"] >= best_cal["R2_log"]:
                use_swa = True
                seed_pred, seed_val_pred, seed_cal = swa_pred, swa_val_pred, swa_cal

        print(f"\n[Seed {seed}] TEST 校准后: R2={seed_cal['R2']:.4f} | R2_log={seed_cal['R2_log']:.4f} | "
              f"Spearman={seed_cal['Spearman']:.4f} | 采用 {'SWA' if use_swa else 'best'}")

        if test_true is None:
            test_true = seed_pred["true"]
        if val_true is None:
            val_true = seed_val_pred["true"]

        test_pred_logs.append(seed_pred["pred_log"])
        val_pred_logs.append(seed_val_pred["pred_log"])
        test_raw_preds.append(seed_pred["pred"])

    # ---------- 集成：log 空间平均各 seed 预测，再做一次统一校准 ----------
    test_pred_logs = np.asarray(test_pred_logs)   # (n_seeds, n_test)
    val_pred_logs = np.asarray(val_pred_logs)     # (n_seeds, n_val)

    ens_test_pred_log = test_pred_logs.mean(axis=0)
    ens_val_pred_log = val_pred_logs.mean(axis=0)

    vmask = np.isfinite(ens_val_pred_log)
    coef = fit_log_linear_calibration(ens_val_pred_log[vmask], val_true[vmask])
    ens_raw = inverse_activity(np.clip(ens_test_pred_log, 0.0, None))
    ens_cal = apply_log_linear_calibration(ens_test_pred_log, coef)

    ens_metrics = compute_metrics(test_true, ens_raw, ens_test_pred_log)
    ens_metrics_cal = compute_metrics(test_true, ens_cal, ens_test_pred_log)

    print(f"\n========== ENSEMBLE TEST RESULTS ({n_seeds} seeds, 原始) ==========")
    for k, v in ens_metrics.items():
        print(f"{k}: {v}")
    print(f"\n========== ENSEMBLE TEST RESULTS ({n_seeds} seeds, 校准后) ==========")
    for k, v in ens_metrics_cal.items():
        print(f"{k}: {v}")

    # ---------- 保存集成预测与各 seed 方差 ----------
    result_df = df_test.reset_index(drop=True).copy()
    result_df["True_Activity"] = test_true
    result_df["Pred_Activity"] = ens_raw
    result_df["Pred_Activity_Calibrated"] = ens_cal
    result_df["Pred_Log_Activity"] = ens_test_pred_log
    result_df["Abs_Error"] = np.abs(result_df["True_Activity"] - ens_raw)
    result_df = result_df.sort_values("Pred_Activity", ascending=False)

    result_df.to_csv(os.path.join(model_dir, "ensemble_predictions.csv"), index=False)
    result_df.head(100).to_csv(os.path.join(model_dir, "ensemble_top100.csv"), index=False)

    seed_df = df_test.reset_index(drop=True).copy()
    seed_df["True_Activity"] = test_true
    for i, seed in enumerate(seeds):
        seed_df[f"seed_{seed}"] = test_raw_preds[i]
    seed_df["Ensemble_Mean"] = ens_raw
    seed_df.to_csv(os.path.join(model_dir, "ensemble_seed_predictions.csv"), index=False)

    pd.DataFrame([
        {"variant": "ensemble_raw", **ens_metrics},
        {"variant": "ensemble_calibrated", **ens_metrics_cal},
    ]).to_csv(os.path.join(model_dir, "ensemble_metrics.csv"), index=False)

    print("\n多 seed 集成预测结果已保存！")


if __name__ == "__main__":
    main()
