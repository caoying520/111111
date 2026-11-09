
# -*- coding: utf-8 -*-
import os
import random
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# ========================
# 1. 参数设置
# ========================
SEED=123
BASE_DIR="/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED"
FEATURE_DIR=os.path.join(BASE_DIR,"Feature")
CSV_PATH=os.path.join(BASE_DIR,"train_Activity_std2.csv")
SPLIT_DIR=os.path.join(BASE_DIR,"split_data_Activaty")
ESM_PATH=os.path.join(FEATURE_DIR,"esm_feats_emb_mask.pkl")
MOL_AMINE_PATH=os.path.join(FEATURE_DIR,"SM_胺.pkl")
MOL_KETONE_PATH=os.path.join(FEATURE_DIR,"SM_酮.pkl")
OUT_DIR=os.path.join(BASE_DIR,"model_activity_regression_rank")

BATCH_SIZE=64
EPOCHS=500
LR=1e-4
WEIGHT_DECAY=1e-4
PATIENCE=60
RANK_WEIGHT=0.1
RANK_MIN_DELTA=0.03
ACTIVE_THRESHOLD=0.03
TOP_K=100
NUM_WORKERS=0

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR,exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

set_seed(SEED)

# ========================
# 2. 数据读取
# ========================
def load_pkl(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def normalize_id(x):
    s=str(x).strip()
    try:
        v=float(s)
        if np.isfinite(v) and v.is_integer():
            return str(int(v))
    except:
        pass
    return s

def read_split_ids(path):
    ids=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            value=line.replace(","," ").split()[0]
            if value.lower() in ["idx","id","index"]:
                continue
            ids.append(normalize_id(value))
    return set(ids)

def transform_activity(y):
    return np.log1p(np.maximum(np.asarray(y,dtype=np.float32),0))

def get_esm_data(esm_embeds,key):
    data=esm_embeds[key]
    if isinstance(data,dict):
        emb=data["emb"]
        mask=data.get("mask",None)
    elif isinstance(data,(tuple,list)):
        emb=data[0]
        mask=data[1] if len(data)>1 else None
    else:
        raise TypeError(f"不支持的ESM特征格式: {type(data)}")
    emb=torch.as_tensor(emb,dtype=torch.float32)
    if emb.ndim==3 and emb.shape[0]==1:
        emb=emb.squeeze(0)
    if emb.ndim!=2:
        raise ValueError(f"ESM embedding维度错误: {key}, {emb.shape}")
    if mask is None:
        mask=torch.ones(emb.shape[0],dtype=torch.long)
    else:
        mask=torch.as_tensor(mask,dtype=torch.long).reshape(-1)
    L=emb.shape[0]
    if mask.shape[0]>L:
        mask=mask[:L]
    elif mask.shape[0]<L:
        mask=F.pad(mask,(0,L-mask.shape[0]),value=0)
    if mask.sum().item()==0:
        raise ValueError(f"ESM mask全部为0: {key}")
    return emb,mask

def get_mol_data(mol_embeds,key):
    data=mol_embeds[key]
    if isinstance(data,dict):
        for k in ["emb","embedding","features","feature"]:
            if k in data:
                data=data[k]
                break
    feat=torch.as_tensor(data,dtype=torch.float32).reshape(-1)
    if feat.numel()!=768:
        raise ValueError(f"MolFormer维度错误: {key}, shape={feat.shape}")
    return feat

df=pd.read_csv(CSV_PATH)
df["idx"]=df["idx"].apply(normalize_id)
df["Name"]=df["Name"].astype(str)
df["SM_胺"]=df["SM_胺"].fillna("").astype(str)
df["SM_酮"]=df["SM_酮"].fillna("").astype(str)
df["Activity"]=pd.to_numeric(df["Activity"],errors="coerce")
df=df.dropna(subset=["Activity"]).reset_index(drop=True)

if (df["Activity"]<0).any():
    raise ValueError("当前代码使用log1p，Activity存在负值，请检查数据。")

# 在完整数据上生成组ID，三个数据集使用统一的底物组合编号
df["group_id"]=pd.factorize(df["SM_胺"]+"|||"+df["SM_酮"])[0]

esm_embeds=load_pkl(ESM_PATH)
mol_embeds1=load_pkl(MOL_AMINE_PATH)
mol_embeds2=load_pkl(MOL_KETONE_PATH)

print(f"ESM={len(esm_embeds)}, MolFormer胺={len(mol_embeds1)}, MolFormer酮={len(mol_embeds2)}")

# 统一特征字典的key
esm_embeds={str(k):v for k,v in esm_embeds.items()}
mol_embeds1={str(k):v for k,v in mol_embeds1.items()}
mol_embeds2={str(k):v for k,v in mol_embeds2.items()}

# ========================
# 3. 读取已经划分好的三个txt
# ========================
train_ids=read_split_ids(os.path.join(SPLIT_DIR,"train.txt"))
val_ids=read_split_ids(os.path.join(SPLIT_DIR,"val.txt"))
test_ids=read_split_ids(os.path.join(SPLIT_DIR,"test.txt"))

train_df=df[df["idx"].isin(train_ids)].reset_index(drop=True)
val_df=df[df["idx"].isin(val_ids)].reset_index(drop=True)
test_df=df[df["idx"].isin(test_ids)].reset_index(drop=True)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if min(len(train_df),len(val_df),len(test_df))==0:
    raise ValueError("某个数据集为空，请检查txt中的idx是否与CSV中的idx对应。")

if set(train_df["idx"]) & set(val_df["idx"]) or set(train_df["idx"]) & set(test_df["idx"]) or set(val_df["idx"]) & set(test_df["idx"]):
    raise ValueError("train、val、test存在重复idx，请检查原始划分文件。")

# ========================
# 4. Dataset定义
# ========================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self,df,esm_embeds,mol_embeds1,mol_embeds2):
        self.df=df.reset_index(drop=True)
        self.idx=self.df["Name"].astype(str).tolist()
        self.smiles1=self.df["SM_胺"].tolist()
        self.smiles2=self.df["SM_酮"].tolist()
        self.labels=pd.to_numeric(self.df["Activity"],errors="coerce").fillna(0).astype(float).to_numpy()
        self.labels_trans=transform_activity(self.labels)
        self.group_ids=self.df["group_id"].to_numpy()
        self.esm_embeds=esm_embeds
        self.mol_embeds1=mol_embeds1
        self.mol_embeds2=mol_embeds2

        for key in self.idx:
            if key not in self.esm_embeds:
                raise KeyError(f"ESM特征不存在: {key}")
        for key in set(self.smiles1):
            if key not in self.mol_embeds1:
                raise KeyError(f"胺MolFormer特征不存在: {key}")
        for key in set(self.smiles2):
            if key not in self.mol_embeds2:
                raise KeyError(f"酮MolFormer特征不存在: {key}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self,i):
        esm_emb,esm_mask=get_esm_data(self.esm_embeds,self.idx[i])
        mol_feat1=get_mol_data(self.mol_embeds1,self.smiles1[i])
        mol_feat2=get_mol_data(self.mol_embeds2,self.smiles2[i])
        return (
            esm_emb,
            esm_mask,
            mol_feat1,
            mol_feat2,
            torch.tensor(self.labels_trans[i],dtype=torch.float32),
            torch.tensor(self.labels[i],dtype=torch.float32),
            torch.tensor(self.group_ids[i],dtype=torch.long)
        )

# ========================
# 5. collate_fn
# ========================
def collate_fn(batch):
    max_len=max(x[0].shape[0] for x in batch)
    embs=[]
    masks=[]
    for x in batch:
        emb=torch.as_tensor(x[0],dtype=torch.float32)
        mask=torch.as_tensor(x[1],dtype=torch.long)
        L=emb.shape[0]
        if mask.shape[0]>L:
            mask=mask[:L]
        elif mask.shape[0]<L:
            mask=F.pad(mask,(0,L-mask.shape[0]),value=0)
        pad_len=max_len-L
        embs.append(F.pad(emb,(0,0,0,pad_len),value=0))
        masks.append(F.pad(mask,(0,pad_len),value=0))

    esm_emb=torch.stack(embs)
    esm_mask=torch.stack(masks)
    mol_amine=torch.stack([x[2] for x in batch])
    mol_ketone=torch.stack([x[3] for x in batch])
    reg_label=torch.stack([x[4] for x in batch])
    raw_activity=torch.stack([x[5] for x in batch])
    group_id=torch.stack([x[6] for x in batch])

    return esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id

# ========================
# 6. Group-aware BatchSampler
# ========================
class GroupBatchSampler(Sampler):
    def __init__(self,group_ids,batch_size,shuffle=True):
        self.group_ids=np.asarray(group_ids)
        self.batch_size=batch_size
        self.shuffle=shuffle

    def __iter__(self):
        groups=np.unique(self.group_ids)
        if self.shuffle:
            np.random.shuffle(groups)

        chunks=[]
        for g in groups:
            indices=np.where(self.group_ids==g)[0].tolist()
            if self.shuffle:
                random.shuffle(indices)
            chunks.extend([indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)])

        if self.shuffle:
            random.shuffle(chunks)

        batches=[]
        current=[]
        for chunk in chunks:
            if current and len(current)+len(chunk)>self.batch_size:
                batches.append(current)
                current=[]
            current.extend(chunk)

        if current:
            batches.append(current)

        if len(batches)>1 and len(batches[-1])==1:
            batches[-2].extend(batches.pop())

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return max(1,int(np.ceil(len(self.group_ids)/self.batch_size)))

# ========================
# 7. DataLoader
# ========================
train_dataset=EnzymeSubstrateDataset(train_df,esm_embeds,mol_embeds1,mol_embeds2)
val_dataset=EnzymeSubstrateDataset(val_df,esm_embeds,mol_embeds1,mol_embeds2)
test_dataset=EnzymeSubstrateDataset(test_df,esm_embeds,mol_embeds1,mol_embeds2)

train_sampler=GroupBatchSampler(train_dataset.group_ids,BATCH_SIZE,shuffle=True)

train_loader=DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ========================
# 8. 原始模型架构
# ========================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self,esm_dim=1280,mol_dim=768,dropout=0.25,hidden_dim=256):
        super().__init__()

        self.esm_proj=nn.Sequential(
            nn.Linear(esm_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256)
        )

        self.mol_proj1=nn.Sequential(
            nn.Linear(mol_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256)
        )

        self.mol_proj2=nn.Sequential(
            nn.Linear(mol_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256)
        )

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.esm_encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )

        self.fc1=nn.Linear(768,hidden_dim)
        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim//2)
        self.bn3=nn.BatchNorm1d(hidden_dim//2)
        self.dropout=nn.Dropout(dropout)

        self.reg_head=nn.Linear(hidden_dim//2,1)

    def forward(self,esm_feat,esm_mask,mol_feat1,mol_feat2):
        esm_feat=self.esm_proj(esm_feat)

        assert esm_feat.shape[:2]==esm_mask.shape, f"ESM={esm_feat.shape}, Mask={esm_mask.shape}"

        esm_mask=esm_mask.bool()
        esm_feat=self.esm_encoder(
            esm_feat,
            src_key_padding_mask=~esm_mask
        )

        mask=esm_mask.unsqueeze(-1).float()
        esm_pooled=(esm_feat*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)

        mol1=self.mol_proj1(mol_feat1)
        mol2=self.mol_proj2(mol_feat2)

        x=torch.cat([esm_pooled,mol1,mol2],dim=-1)

        x=F.relu(self.bn1(self.fc1(x)))
        residual=x
        x=self.dropout(F.relu(self.bn2(self.fc2(x))))+residual
        x=self.dropout(F.relu(self.bn3(self.fc3(x))))

        return self.reg_head(x).squeeze(-1)

# ========================
# 9. Pairwise Ranking Loss
# ========================
def pairwise_ranking_loss(pred,raw_activity,group_id,min_delta=0.03):
    pred=pred.reshape(-1)
    raw_activity=raw_activity.reshape(-1)
    group_id=group_id.reshape(-1)

    group_losses=[]

    for g in torch.unique(group_id):
        idx=torch.where(group_id==g)[0]

        if len(idx)<2:
            continue

        pair=torch.triu_indices(
            len(idx),
            len(idx),
            offset=1,
            device=pred.device
        )

        i=idx[pair[0]]
        j=idx[pair[1]]

        true_diff=raw_activity[i]-raw_activity[j]
        valid=true_diff.abs()>=min_delta

        if valid.any():
            sign=torch.sign(true_diff[valid])
            pred_diff=pred[i[valid]]-pred[j[valid]]
            group_losses.append(F.softplus(-sign*pred_diff).mean())

    if not group_losses:
        return pred.sum()*0.0

    return torch.stack(group_losses).mean()

# ========================
# 10. 评价指标
# ========================
def safe_corr(func,y_true,y_pred):
    if len(y_true)<2 or np.std(y_true)==0 or np.std(y_pred)==0:
        return np.nan
    try:
        return float(func(y_true,y_pred)[0])
    except:
        return np.nan

def calculate_metrics(y_true,y_pred,top_k=100):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)

    k=min(top_k,len(y_true))

    mae=mean_absolute_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r2=r2_score(y_true,y_pred) if len(y_true)>1 else np.nan
    pearson=safe_corr(pearsonr,y_true,y_pred)
    spearman=safe_corr(spearmanr,y_true,y_pred)

    active_mask=y_true>=ACTIVE_THRESHOLD
    active_spearman=safe_corr(
        spearmanr,
        y_true[active_mask],
        y_pred[active_mask]
    ) if active_mask.sum()>=2 else np.nan

    ndcg=ndcg_score(
        y_true.reshape(1,-1),
        y_pred.reshape(1,-1),
        k=k
    ) if np.any(y_true>0) else 0.0

    pred_top=np.argsort(-y_pred)[:k]
    true_top=np.argsort(-y_true)[:k]

    overlap=len(set(pred_top)&set(true_top))
    overlap_ratio=overlap/k if k>0 else np.nan
    top100_true_mean=float(y_true[pred_top].mean()) if k>0 else np.nan
    true_top_mean=float(y_true[true_top].mean()) if k>0 else np.nan

    return {
        "MAE":float(mae),
        "RMSE":float(rmse),
        "R2":float(r2),
        "Pearson":pearson,
        "Spearman":spearman,
        "Active_Spearman":active_spearman,
        "NDCG@100":float(ndcg),
        "Top100_Overlap":float(overlap_ratio),
        "Top100_True_Mean":top100_true_mean,
        "True_Top100_Mean":true_top_mean,
        "Active_Count":int(active_mask.sum())
    }

# ========================
# 11. 验证/测试
# ========================
@torch.no_grad()
def evaluate(model,loader,criterion):
    model.eval()
    total_loss=0.0
    total_num=0
    preds=[]
    labels=[]

    for esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id in loader:
        esm_emb=esm_emb.to(DEVICE,non_blocking=True)
        esm_mask=esm_mask.to(DEVICE,non_blocking=True)
        mol_amine=mol_amine.to(DEVICE,non_blocking=True)
        mol_ketone=mol_ketone.to(DEVICE,non_blocking=True)
        reg_label=reg_label.to(DEVICE,non_blocking=True)

        pred=model(esm_emb,esm_mask,mol_amine,mol_ketone)
        loss=criterion(pred,reg_label)

        total_loss+=loss.item()*len(reg_label)
        total_num+=len(reg_label)

        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(raw_activity.numpy().tolist())

    pred_activity=np.maximum(np.expm1(np.asarray(preds)),0)
    y_true=np.asarray(labels)

    metrics=calculate_metrics(y_true,pred_activity)
    metrics["Loss"]=total_loss/max(total_num,1)

    return metrics,pred_activity,y_true

# ========================
# 12. 训练
# ========================
def train_model():
    model=EnzymeSubstrateRegressor().to(DEVICE)

    criterion=nn.SmoothL1Loss(beta=0.5)

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    best_score=-float("inf")
    best_epoch=0
    patience_counter=0
    history=[]

    print("\n开始回归 + Pairwise Ranking训练")
    print(f"Device={DEVICE}, Batch Size={BATCH_SIZE}, Rank Weight={RANK_WEIGHT}")

    for epoch in range(1,EPOCHS+1):
        model.train()

        train_reg_sum=0.0
        train_rank_sum=0.0
        train_total_sum=0.0
        train_num=0
        valid_rank_batches=0

        pbar=tqdm(train_loader,desc=f"Epoch {epoch}/{EPOCHS}")

        for esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id in pbar:
            esm_emb=esm_emb.to(DEVICE,non_blocking=True)
            esm_mask=esm_mask.to(DEVICE,non_blocking=True)
            mol_amine=mol_amine.to(DEVICE,non_blocking=True)
            mol_ketone=mol_ketone.to(DEVICE,non_blocking=True)
            reg_label=reg_label.to(DEVICE,non_blocking=True)
            raw_activity=raw_activity.to(DEVICE,non_blocking=True)
            group_id=group_id.to(DEVICE,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred=model(esm_emb,esm_mask,mol_amine,mol_ketone)

            reg_loss=criterion(pred,reg_label)

            rank_loss=pairwise_ranking_loss(
                pred,
                raw_activity,
                group_id,
                min_delta=RANK_MIN_DELTA
            )

            loss=reg_loss+RANK_WEIGHT*rank_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()

            bs=len(reg_label)
            train_reg_sum+=reg_loss.item()*bs
            train_rank_sum+=rank_loss.item()*bs
            train_total_sum+=loss.item()*bs
            train_num+=bs

            if rank_loss.item()>0:
                valid_rank_batches+=1

            pbar.set_postfix(
                Reg=f"{reg_loss.item():.4f}",
                Rank=f"{rank_loss.item():.4f}"
            )

        train_reg=train_reg_sum/max(train_num,1)
        train_rank=train_rank_sum/max(train_num,1)
        train_total=train_total_sum/max(train_num,1)

        val_metrics,_,_=evaluate(model,val_loader,criterion)

        val_ndcg=val_metrics["NDCG@100"]
        scheduler.step(val_ndcg)

        current_lr=optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss={train_total:.4f} | "
            f"Reg Loss={train_reg:.4f} | "
            f"Rank Loss={train_rank:.4f} | "
            f"Valid Rank Batches={valid_rank_batches} | "
            f"Val Loss={val_metrics['Loss']:.4f} | "
            f"MAE={val_metrics['MAE']:.4f} | "
            f"RMSE={val_metrics['RMSE']:.4f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"Spearman={val_metrics['Spearman']:.4f} | "
            f"Active Spearman={val_metrics['Active_Spearman']:.4f} | "
            f"NDCG@100={val_ndcg:.4f} | "
            f"Top100 Overlap={val_metrics['Top100_Overlap']:.4f} | "
            f"LR={current_lr:.7f}"
        )

        history.append({
            "Epoch":epoch,
            "Train_Loss":train_total,
            "Train_Reg_Loss":train_reg,
            "Train_Rank_Loss":train_rank,
            "Valid_Rank_Batches":valid_rank_batches,
            **{f"Val_{k}":v for k,v in val_metrics.items()},
            "LR":current_lr
        })

        if np.isfinite(val_ndcg) and val_ndcg>best_score:
            best_score=val_ndcg
            best_epoch=epoch
            patience_counter=0

            torch.save({
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "best_val_ndcg":best_score,
                "val_metrics":val_metrics
            },os.path.join(OUT_DIR,"best_model.pth"))

            print(f"保存最佳模型: Epoch={epoch}, NDCG@100={best_score:.4f}")

        else:
            patience_counter+=1

        if patience_counter>=PATIENCE:
            print(f"\nEarly Stopping: Epoch={epoch}, Best Epoch={best_epoch}, Best NDCG@100={best_score:.4f}")
            break

        pd.DataFrame(history).to_csv(
            os.path.join(OUT_DIR,"training_history.csv"),
            index=False
        )

    # ========================
    # 13. 加载最佳模型并测试
    # ========================
    checkpoint=torch.load(
        os.path.join(OUT_DIR,"best_model.pth"),
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics,test_pred,test_true=evaluate(model,test_loader,criterion)

    print("\n========== Test Results ==========")
    for k,v in test_metrics.items():
        print(f"{k}: {v}")

    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(OUT_DIR,"test_metrics.csv"),
        index=False
    )

    # ========================
    # 14. 保存测试集预测结果
    # ========================
    test_result=test_df.copy()
    test_result["True_Activity"]=test_true
    test_result["Pred_Activity"]=test_pred
    test_result["Pred_Rank"]=test_result["Pred_Activity"].rank(
        ascending=False,
        method="min"
    ).astype(int)
    test_result["True_Rank"]=test_result["True_Activity"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    test_result.to_csv(
        os.path.join(OUT_DIR,"test_predictions.csv"),
        index=False
    )

    test_top100=test_result.sort_values(
        "Pred_Activity",
        ascending=False
    ).head(TOP_K)

    test_top100.to_csv(
        os.path.join(OUT_DIR,"test_top100.csv"),
        index=False
    )

    with open(os.path.join(OUT_DIR,"config.json"),"w",encoding="utf-8") as f:
        json.dump({
            "SEED":SEED,
            "BATCH_SIZE":BATCH_SIZE,
            "LR":LR,
            "WEIGHT_DECAY":WEIGHT_DECAY,
            "RANK_WEIGHT":RANK_WEIGHT,
            "RANK_MIN_DELTA":RANK_MIN_DELTA,
            "ACTIVE_THRESHOLD":ACTIVE_THRESHOLD,
            "TOP_K":TOP_K,
            "BEST_EPOCH":best_epoch,
            "BEST_VAL_NDCG":best_score
        },f,ensure_ascii=False,indent=4)

    print(f"\n训练完成，最佳Epoch={best_epoch}")
    print(f"最佳验证集NDCG@100={best_score:.4f}")
    print(f"结果保存目录: {OUT_DIR}")

if __name__=="__main__":
    train_model()
