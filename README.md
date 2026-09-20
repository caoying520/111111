# -*- coding: utf-8 -*-
#np.set_printoptions(threshold=np.inf)
import os
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import heapq
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# # 禁用 Nested Tensor 优化
# torch.backends.cuda.enable_nested_tensor(False)
# # 也可以同时禁用 Flash Attention 以确保行为完全一致（可选）
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# ========================
# 0️⃣ 固定随机种子 + 确定性设置
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# ========================
# 1️⃣ 四分类映射
# ========================
def map_value_to_label(x):
    # 0:[0,0.3), 1:[0.3,0.5), 2:[0.5,0.7), 3:[0.7,0.9), 4:[0.9,1.0]
    if 0.0 <= x < 0.3:
        return 0
    elif 0.3 <= x < 0.5:
        return 1
    elif 0.5 <= x < 0.7:
        return 2
    elif 0.7 <= x < 0.9:
        return 3
    elif 0.9 <= x <= 1.0:
        return 4
    else:
        print(f"Warning: {x} is out of range [0,1]")
        return None

OBJ = "Conv_5cls"

# ========================
# 2️⃣ Dataset（加入 SM_norm / Enz_norm）
# ========================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self, df, esm_embeds, mol_embeds,
                 esm_key_col="Name", smiles_col="SM",
                 sm_norm_col="SM_norm", enz_norm_col="Enz_norm"):
        self.df = df.reset_index(drop=True)

        # ESM key（用于 esm_embeds 查找）
        self.esm_keys = self.df[esm_key_col].astype(str).tolist()
        self.smiles = self.df[smiles_col].tolist()
        self.labels = self.df[OBJ].astype(int).tolist()

        self.esm_embeds = esm_embeds
        self.mol_embeds = mol_embeds

        if sm_norm_col not in self.df.columns or enz_norm_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{sm_norm_col}' and '{enz_norm_col}' columns.")
        self.sm_norm = self.df[sm_norm_col].astype(float).tolist()
        self.enz_norm = self.df[enz_norm_col].astype(float).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        key = self.esm_keys[i]                 #从 df 表格的第 i 行拿到了一个字符串名字
        esm_data = self.esm_embeds[key]        #蛋白质维度 (esm_emb, esm_mask)：从 esm_embeds 字典中按 Name 索引。
        esm_emb = esm_data["emb"].float()      # [B,L, 1280]     2D 矩阵D 矩阵 [L, 1280]，代表序列中每个位置的信息。ESM 处理脚本通常直接把结果存成了 Tensor。所以这里不需要用 torch.tensor() 转换，只需要用 .float() 或 .long() 确保精度正确
        esm_mask = esm_data["mask"].long()     # [B,L]

        mol_feat = torch.tensor(self.mol_embeds[self.smiles[i]], dtype=torch.float32)     #[B, 768]   1D 向量（768 维），没有 Mask。小分子维度从 mol_embeds 字典中按 SMILES 字符串索引。原始的 mol_embeds 字典里存的往往是 NumPy 数组或 Python 列表（取决于提取脚本）

        # 浓度：归一化后的标量 -> shape [1] 。 直接从 Dataframe 的列中提取已经归一化后的浓度数值。
        sm_load = torch.tensor([self.sm_norm[i]], dtype=torch.float32)          #[B, 1]    加上 [] 后，它变成了维度为 [1] 的向量，如果不加 []，转换后维度是 0
        enz_load = torch.tensor([self.enz_norm[i]], dtype=torch.float32)

        label = torch.tensor(self.labels[i], dtype=torch.long)             #[B] 0~4 的分类标签。不需要额外的维度（不需要加 []），直接是一个标量即可
        return esm_emb, esm_mask, mol_feat, sm_load, enz_load, label

# ========================
# 3️⃣ 模型（加入两个浓度投影：1 -> 64 -> 128）
# ========================
# class EnzymeSubstrateClassifier(nn.Module):
#     def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.15,
#                  hidden_dim=256, num_classes=5):
#         super().__init__()

#         # 1. 基础投影层 (维持原样)
#         self.esm_proj = nn.Sequential(
#             nn.Linear(esm_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#         )
#         self.mol_proj = nn.Sequential(
#             nn.Linear(mol_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#         )
        
#         # 2. 酶序列自注意力编码 (维持原样)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=256, nhead=8, dim_feedforward=512,
#             dropout=dropout, batch_first=True
#         )
#         self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         # 🌟 3. 新增：Cross-Attention 模块
#         # 让底物 (256维) 去注意 酶序列 (L x 256维)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=256, 
#             num_heads=8, 
#             dropout=dropout, 
#             batch_first=True
#         )
#         # 用于 Cross-Attention 后的层归一化
#         self.ln_cross = nn.LayerNorm(256)
#         self.post_attn_dropout = nn.Dropout(dropout)

#         # 4. 浓度特征投影 (维持原样)
#         self.sm_proj = nn.Sequential(
#             nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128),
#         )
#         self.enz_proj = nn.Sequential(
#             nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128),
#         )

#         # 5. 全连接分类层 (输入维度依然是 768)
#         # [ESM全局(256) + 交互后底物(256) + 浓度SM(128) + 浓度Enz(128)] = 768
#         input_dim = 256 + 256 + 128 + 128
#         self.fc_blocks = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
    
#     def forward(self, esm_feat, esm_mask, mol_feat, sm_load, enz_load):
#         # 获取特征实际的序列长度 (例如 400)
#         actual_seq_len = esm_feat.size(1)
#         print(f"Debug: actual_seq_len = {actual_seq_len}, esm_feat.shape = {esm_feat.shape}, esm_mask.shape = {esm_mask.shape}")
        
#         # 🌟 关键修正：确保 mask 的长度和特征一致
#         # 只取前 actual_seq_len 个位置
#         valid_esm_mask = esm_mask[:, :actual_seq_len]
        
#         # A. 酶分支处理 [B, L, 256]
#         esm_feat = self.esm_proj(esm_feat)
        
#         # src_mask 用于 Transformer 的 Padding Mask (True 代表要遮盖的位置)
#         src_mask = (valid_esm_mask == 0) 
        
#         # 传入对齐后的 src_mask
#         esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask) 

#         # B. 底物分支处理 [B, 256]
#         mol_feat_proj = self.mol_proj(mol_feat)

#         # C. 执行 Cross-Attention 交互
#         query = mol_feat_proj.unsqueeze(1) 
#         key = esm_feat
#         value = esm_feat
        
#         # 🌟 关键修正：这里也要传入对齐后的 src_mask
#         attn_out, _ = self.cross_attn(
#             query=query, 
#             key=key, 
#             value=value, 
#             key_padding_mask=src_mask
#         )
        
#         # 残差连接 + 层归一化
#         # (注：原代码这里写了两次 ln_cross，建议精简如下)
#         mol_refined = attn_out.squeeze(1) + mol_feat_proj
#         mol_refined = self.ln_cross(mol_refined)
#         mol_refined = self.post_attn_dropout(mol_refined)

#         # D. 酶的全局池化 (也要用对齐后的 mask)
#         mask_expanded = valid_esm_mask.unsqueeze(-1).float() # [B, L, 1]
#         denom = mask_expanded.sum(dim=1).clamp(min=1e-8)
#         esm_pooled = (esm_feat * mask_expanded).sum(dim=1) / denom

#         # E. 浓度投影
#         sm_p = self.sm_proj(sm_load)
#         enz_p = self.enz_proj(enz_load)

#         # F. 最终拼接与预测
#         x = torch.cat([esm_pooled, enz_p, mol_refined, sm_p], dim=-1) # [B, 768]
        
#         return self.fc_blocks(x)

class EnzymeSubstrateClassifier(nn.Module):
    def __init__(self, esm_dim=1280, mol_dim=768, dropout=0.1,
                 hidden_dim=256, num_classes=5):
        super().__init__()

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        
        ##仅仅靠 ESM 提取的静态特征不够，定义了一个 2 层、8 个注意力头（nhead=8）的 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.esm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

         # 让底物 (256维) 去注意 酶序列 (L x 256维)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        # 用于 Cross-Attention 后的层归一化
        self.ln_cross = nn.LayerNorm(256)

        # 两个浓度特征投影 单一的标量数值（1维浓度）通过 (1 -> 64 -> 128) 升维
        self.sm_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        )
        self.enz_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        )

        # 输入维度：ESM(256) + Mol(256) + SM(128) + Enz(128) = 768
        input_dim = 256 + 256 + 128 + 128

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.2)


#原架构： 先把酶序列做 Mean Pooling 变成一个向量，再和小分子向量拼接。这就像是把一本书（序列）压缩成一句话，再和另一个单词（分子）拼在一起。
#新架构： 底物分子像一个“扫描仪”，通过 Cross-Attention 逐个扫描酶序列中的氨基酸。如果某个氨基酸对底物很重要（比如活性中心），Attention 权重就会很高。最终得到的 mol_refined 是专门针对这个酶优化过的底物特征。
    
    def forward(self, esm_feat, esm_mask, mol_feat, sm_load, enz_load):
        # esm_feat: [B, L, 1280], esm_mask: [B, L]
        esm_feat = self.esm_proj(esm_feat)  # [B, L, 256]

        src_mask = (esm_mask == 0)  # True where padding
        esm_feat = self.esm_encoder(esm_feat, src_key_padding_mask=src_mask)  # [B, L, 256]

        # 注意：这里 esm_feat 输入是 400，但由于 PyTorch 内部优化， 如果当前 Batch 最长序列只有 388，输出 esm_feat 可能会变成 [B, 388, 256]
        # --- ✨ 核心修复：强行补回 400 位 ✨ ---
        # curr_len = esm_feat.shape[1]
        # if curr_len < 400:
        #     diff = 400 - curr_len
        #     # 创建一个全零的补丁，维度为 [B, 缺少的长度, 256]
        #     padding = torch.zeros(
        #         (esm_feat.shape[0], diff, esm_feat.shape[2]), 
        #         device=esm_feat.device, 
        #         dtype=esm_feat.dtype
        #     )
        #     # 把补丁拼接到末尾，强行变回 [B, 400, 256]
        #     esm_feat = torch.cat([esm_feat, padding], dim=1)
       
        actual_seq_len = esm_feat.size(1)
        # 同步裁剪 mask，确保它的长度和 esm_feat 的第二维完全一致.这样无论 PyTorch 怎么缩减长度，相乘时都不会报错
        esm_mask = esm_mask[:, :actual_seq_len]

        mask_expanded = esm_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask_expanded.sum(dim=1).clamp(min=1e-8)  # [B, 1]
        esm_pooled = (esm_feat * mask_expanded).sum(dim=1) / denom  # [B, 256]     #Masked Mean Pooling (过滤Padding并求有效均值)

        mol_proj = self.mol_proj(mol_feat)  # [B, 256]
        sm_proj = self.sm_proj(sm_load)     # [B, 128]
        enz_proj = self.enz_proj(enz_load)  # [B, 128]

        
        x = torch.cat([esm_pooled, enz_proj, mol_proj, sm_proj], dim=-1)  # [B, 768]

        x = F.relu(self.bn1(self.fc1(x)))
        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # [B, num_classes]
        return x

# ========================
# 4️⃣ 评估函数（加入浓度输入）
# ========================
def evaluate_cls_metrics_loss(model, loader, device, criterion=None, num_classes=5):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in loader:
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)
            y = y.to(device)

            logits = model(esm_feat, esm_mask, mol_feat, sm_load, enz_load)
            probs = F.softmax(logits, dim=1)

            if criterion is not None:
                total_loss += criterion(logits, y).item()

            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = np.argmax(all_probs, axis=1)
    acc = (preds == all_labels).mean()
    avg_loss = total_loss / max(len(loader), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")          #衡量的是模型区分不同类别的能力，衡量的是模型在召回（找出正样本）的同时，保持高精确度（找得准）的能力
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(np.eye(num_classes)[all_labels], all_probs, average="macro")     #衡量的是模型在召回（找出正样本）的同时，保持高精确度（找得准）的能力。是对 5 个类别的 AP 取平均值
    except ValueError:
        ap = float("nan")

    return avg_loss, acc, auc, ap

# ========================
# 5️⃣ 训练函数（按 AP 保存 Top-5 + best）
# ========================
def train_model_cls_top5_loss(train_loader, val_loader, model, optimizer, device,
                              epochs=200, num_classes=5, model_dir="model_5cls_conc"):
    criterion = nn.CrossEntropyLoss()
    top_models = []
    best_ap = -np.inf
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        # for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in tqdm(
        #     train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        # ):
        for esm_feat, esm_mask, mol_feat, sm_load, enz_load, y in train_loader:
            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat = mol_feat.to(device)
            sm_load = sm_load.to(device)
            enz_load = enz_load.to(device)
            y = y.to(device)

            logits = model(esm_feat, esm_mask, mol_feat, sm_load, enz_load)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            #新增梯度裁剪，防止梯度爆炸
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss, train_acc, train_auc, train_ap = evaluate_cls_metrics_loss(
            model, train_loader, device, criterion, num_classes
        )
        val_loss, val_acc, val_auc, val_ap = evaluate_cls_metrics_loss(
            model, val_loader, device, criterion, num_classes
        )

        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Train: Loss={train_loss:.4f} | ACC={train_acc:.4f} | AUC={train_auc:.4f} | AP={train_ap:.4f} || "
            f"Val: Loss={val_loss:.4f} | ACC={val_acc:.4f} | AUC={val_auc:.4f} | AP={val_ap:.4f}"
        )

        model_path = os.path.join(model_dir, f"epoch_{epoch+1:04d}_AP_{val_ap:.4f}.pth")

        #如果目前保存的模型不足 5 个，不管当前模型表现如何，直接塞进排行榜，并物理保存到硬盘
        if len(top_models) < 5:            
            heapq.heappush(top_models, (val_ap, model_path))       #heapq 会把最小的元素放在最前面
            torch.save(model.state_dict(), model_path)
        elif val_ap > top_models[0][0]:                            # 如果当前 AP 大于排行榜里的最小值
            worst_ap, worst_path = heapq.heappop(top_models)       # 踢出原来的最后一名
            heapq.heappush(top_models, (val_ap, model_path))       # 将新挑战者加入排行榜
            torch.save(model.state_dict(), model_path)
            try:
                os.remove(worst_path)                              # 物理删除被踢出的模型文件，节省空间
            except FileNotFoundError:
                pass

        if val_ap > best_ap:                                       #best_model.pth
            best_ap = val_ap
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

    print("\nTop-5 validation models (by AP):")
    for ap, path in sorted(top_models, key=lambda x: -x[0]):
        print(f"  AP={ap:.4f} | {path}")

# ========================
# 6️⃣ 主程序
# ========================
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------- 读数据 ----------
    # df = pd.read_csv("/home/qian.xu/enzyme/data/out/merged_KRD_S2.csv")
    df = pd.read_csv("/data/caoying/酶/enzyme_交接20260609/enzyme_train/data/酶筛/merged_KRD_S2_KRD_M_add.csv")


    # 分类标签
    df[OBJ] = df["Conv"].apply(map_value_to_label)
    df = df.dropna(subset=[OBJ])

    # split key（用于 train/val/test 切分）
    df["idx"] = df["idx"].astype(str)

    # ---------- 读 split ----------每一行存放一个唯一的索引 ID（idx）
    split_dir = "/data/caoying/酶/enzyme_交接20260609/enzyme_train/data/酶筛/splits/add"
    with open(os.path.join(split_dir, "train_S2_KRD_M_50%_add.txt")) as f:
        train_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "val_S2_KRD_M_50%_5%_add.txt")) as f:
        val_idx = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "test_add.txt")) as f:
        test_idx = [line.strip() for line in f if line.strip()]

    df_train = df[df["idx"].isin(train_idx)].reset_index(drop=True)
    df_val = df[df["idx"].isin(val_idx)].reset_index(drop=True)
    df_test = df[df["idx"].isin(test_idx)].reset_index(drop=True)

    print(f"📂 数据集大小: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ---------- 浓度 log+zscore：只用训练集统计量 ----------
    # 你数据里浓度列名大概率就是下面两个；如果不一样，改这里即可
    SM_COL = "SM loading (g/L)"
    ENZ_COL = "Enzyme loading (g/L)"

    for d in [df_train, df_val, df_test]:
        d[SM_COL] = pd.to_numeric(d[SM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)   #把所有的空值填为 0.0，clip(lower=0.0)：确保浓度没有负数
        d[ENZ_COL] = pd.to_numeric(d[ENZ_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        d["SM_log"] = np.log1p(d[SM_COL].astype(float))             #对数变换
        d["Enz_log"] = np.log1p(d[ENZ_COL].astype(float))

    #计算了训练集的均值和标准差，模型在训练时不能“偷看”验证集或测试集的分布信息
    sm_mean = df_train["SM_log"].mean()
    sm_std = df_train["SM_log"].std() + 1e-12         #1e-12 是为了防止标准差为 0 时出现除以 0 的错误
    enz_mean = df_train["Enz_log"].mean()
    enz_std = df_train["Enz_log"].std() + 1e-12
    
    #标准化，将数据转化为均值为 0，标准差为 1 的标准分布
    for d in [df_train, df_val, df_test]:
        d["SM_norm"] = (d["SM_log"] - sm_mean) / sm_std
        d["Enz_norm"] = (d["Enz_log"] - enz_mean) / enz_std

    print("SM_log mean/std (train):", sm_mean, sm_std)
    print("Enz_log mean/std (train):", enz_mean, enz_std)

    # ---------- 加载特征 ----------
    esm_pkl = "/data/caoying/酶/enzyme_交接20260609/esm_feats_emb_mask.pkl"
    mol_pkl = "/data/caoying/酶/enzyme_交接20260609/SM.pkl"
    with open(esm_pkl, "rb") as f:
        esm_dict = pickle.load(f)         #emb与mask长度都是400
    with open(mol_pkl, "rb") as f:
        mol_dict = pickle.load(f)
    print(f"✅ 已加载 ESM 特征: {len(esm_dict)} 条 | MolFormer 特征: {len(mol_dict)} 条")

    # ---------- Dataset & Loader ----------
    # esm_key_col 默认用 "name"：必须和 esm_dict 的 key 对得上
    train_set = EnzymeSubstrateDataset(df_train, esm_dict, mol_dict, esm_key_col="Name", smiles_col="SM")
    val_set = EnzymeSubstrateDataset(df_val, esm_dict, mol_dict, esm_key_col="Name", smiles_col="SM")
    test_set = EnzymeSubstrateDataset(df_test, esm_dict, mol_dict, esm_key_col="Name", smiles_col="SM")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # ---------- 模型 ----------
    model_dir = "3model_5cls_krd_s2_conc_55%"
    model = EnzymeSubstrateClassifier(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-5)  #原本e-4, 1e-5

    train_model_cls_top5_loss(
        train_loader, val_loader, model, optimizer, device,
        epochs=100, num_classes=5, model_dir=model_dir
    )

    # ---------- 测试 ----------
    best_model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_auc, test_ap = evaluate_cls_metrics_loss(
        model, test_loader, device, criterion=nn.CrossEntropyLoss(), num_classes=5
    )
    print(f"\n🧪 Test | Loss={test_loss:.4f} | ACC={test_acc:.4f} | AUC={test_auc:.4f} | AP={test_ap:.4f}")

if __name__ == "__main__":
    main()
