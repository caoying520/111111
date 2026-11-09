
# -*- coding: utf-8 -*-
import os, random, pickle, copy, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== 1. 参数设置 ====================
SEED=123
BASE_DIR="/data/AIPD/R2/enzyme_交接20260609/enzyme_train/IRED"
FEATURE_DIR=os.path.join(BASE_DIR,"Feature")
CSV_PATH=os.path.join(BASE_DIR,"train_Activity_std2.csv")
SPLIT_DIR=os.path.join(BASE_DIR,"split_data_Activaty")
ESM_PATH=os.path.join(FEATURE_DIR,"esm_feats_emb_mask.pkl")
MOL_AMINE_PATH=os.path.join(FEATURE_DIR,"SM_胺.pkl")
MOL_KETONE_PATH=os.path.join(FEATURE_DIR,"SM_酮.pkl")
OUT_DIR=os.path.join(BASE_DIR,"model_activity_regression_rank")
os.makedirs(OUT_DIR,exist_ok=True)
BATCH_SIZE=256
EPOCHS=500
LR=1e-4
WEIGHT_DECAY=1e-4
PATIENCE=60
RANK_WEIGHT=0.1
RANK_MIN_DELTA=0.03
ACTIVE_THRESHOLD=0.03
TOP_K=100
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def load_pkl(path):
    with open(path,"rb") as f: return pickle.load(f)

def get_esm_feature(value):
    if isinstance(value,dict):
        emb=value.get("emb",value.get("embedding",value.get("features")))
        mask=value.get("mask",value.get("attention_mask"))
    elif isinstance(value,(tuple,list)) and len(value)>=2:
        emb,mask=value[0],value[1]
    else:
        emb,mask=value,None
    emb=np.asarray(emb,dtype=np.float32)
    if emb.ndim!=2: raise ValueError(f"ESM embedding维度错误: {emb.shape}")
    if mask is None: mask=np.ones(emb.shape[0],dtype=np.bool_)
    else: mask=np.asarray(mask).reshape(-1).astype(bool)
    if len(mask)!=len(emb): raise ValueError(f"ESM mask与embedding长度不一致: {mask.shape}, {emb.shape}")
    return emb,mask

def get_mol_feature(value):
    if isinstance(value,dict):
        value=value.get("features",value.get("embedding",value.get("emb",value)))
    arr=np.asarray(value,dtype=np.float32).reshape(-1)
    if arr.shape[0]!=768: raise ValueError(f"MolFormer特征维度应为768，实际为{arr.shape}")
    return arr

# ==================== 2. 读取数据和原有划分 ====================
set_seed(SEED)
print("Using device:",DEVICE)
df=pd.read_csv(CSV_PATH)
df["idx"]=df["idx"].astype(str).str.strip()
df["Name"]=df["Name"].astype(str).str.strip()
df["SM_胺"]=df["SM_胺"].astype(str).str.strip()
df["SM_酮"]=df["SM_酮"].astype(str).str.strip()
df["Activity"]=pd.to_numeric(df["Activity"],errors="raise").astype(np.float32)
if (df["Activity"]<0).any(): raise ValueError("当前代码使用log1p，Activity必须全部非负")
df["group_id"]=pd.factorize(df["SM_胺"]+"|||"+df["SM_酮"])[0].astype(np.int64)

def read_split(path):
    with open(path,"r") as f: return [x.strip() for x in f if x.strip()]
train_idx=read_split(os.path.join(SPLIT_DIR,"train.txt"))
val_idx=read_split(os.path.join(SPLIT_DIR,"val.txt"))
test_idx=read_split(os.path.join(SPLIT_DIR,"test.txt"))
df_train=df[df["idx"].isin(train_idx)].reset_index(drop=True)
df_val=df[df["idx"].isin(val_idx)].reset_index(drop=True)
df_test=df[df["idx"].isin(test_idx)].reset_index(drop=True)
if min(len(df_train),len(df_val),len(df_test))==0: raise ValueError("train/val/test有空数据集，请检查txt中的idx是否与CSV一致")
print(f"Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
for name,data in [("Train",df_train),("Val",df_val),("Test",df_test)]:
    print(f"{name} Activity分布:\n{data['Activity'].describe()}")

# ==================== 3. 读取特征 ====================
esm_dict=load_pkl(ESM_PATH)
mol_amine_dict=load_pkl(MOL_AMINE_PATH)
mol_ketone_dict=load_pkl(MOL_KETONE_PATH)
print(f"ESM={len(esm_dict)}, MolFormer胺={len(mol_amine_dict)}, MolFormer酮={len(mol_ketone_dict)}")

# ==================== 4. Dataset ====================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self,data,esm_dict,mol_amine_dict,mol_ketone_dict):
        self.df=data.reset_index(drop=True)
        self.esm_dict=esm_dict
        self.mol_amine_dict=mol_amine_dict
        self.mol_ketone_dict=mol_ketone_dict
        self.items=[]
        for _,row in self.df.iterrows():
            name=str(row["Name"]).strip()
            sm_amine=str(row["SM_胺"]).strip()
            sm_ketone=str(row["SM_酮"]).strip()
            if name not in esm_dict: raise KeyError(f"ESM特征中找不到Name: {name}")
            if sm_amine not in mol_amine_dict: raise KeyError(f"胺MolFormer特征中找不到底物: {sm_amine}")
            if sm_ketone not in mol_ketone_dict: raise KeyError(f"酮MolFormer特征中找不到底物: {sm_ketone}")
            emb,mask=get_esm_feature(esm_dict[name])
            amine=get_mol_feature(mol_amine_dict[sm_amine])
            ketone=get_mol_feature(mol_ketone_dict[sm_ketone])
            activity=float(row["Activity"])
            self.items.append((emb,mask,amine,ketone,np.float32(np.log1p(activity)),np.float32(activity),int(row["group_id"])))
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        emb,mask,amine,ketone,reg_label,raw_activity,group_id=self.items[idx]
        return emb,mask,amine,ketone,reg_label,raw_activity,group_id

def collate_fn(batch):
    max_len=max(x[0].shape[0] for x in batch)
    batch_size=len(batch)
    esm_emb=torch.zeros(batch_size,max_len,1280,dtype=torch.float32)
    esm_mask=torch.zeros(batch_size,max_len,dtype=torch.bool)
    for i,(emb,mask,*_) in enumerate(batch):
        length=len(emb)
        esm_emb[i,:length]=torch.as_tensor(emb,dtype=torch.float32)
        esm_mask[i,:length]=torch.as_tensor(mask,dtype=torch.bool)
    mol_amine=torch.tensor(np.stack([x[2] for x in batch]),dtype=torch.float32)
    mol_ketone=torch.tensor(np.stack([x[3] for x in batch]),dtype=torch.float32)
    reg_label=torch.tensor([x[4] for x in batch],dtype=torch.float32)
    raw_activity=torch.tensor([x[5] for x in batch],dtype=torch.float32)
    group_id=torch.tensor([x[6] for x in batch],dtype=torch.long)
    return esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id

train_set=EnzymeSubstrateDataset(df_train,esm_dict,mol_amine_dict,mol_ketone_dict)
val_set=EnzymeSubstrateDataset(df_val,esm_dict,mol_amine_dict,mol_ketone_dict)
test_set=EnzymeSubstrateDataset(df_test,esm_dict,mol_amine_dict,mol_ketone_dict)
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=(len(train_set)%BATCH_SIZE==1),collate_fn=collate_fn,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn,num_workers=0,pin_memory=True)

# ==================== 5. 模型：ESM + 双MolFormer + Transformer + 融合回归 ====================
class EnzymeSubstrateRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_proj=nn.Sequential(nn.Linear(1280,512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,256),nn.GELU())
        encoder_layer=nn.TransformerEncoderLayer(d_model=256,nhead=8,dim_feedforward=512,dropout=0.2,batch_first=True,activation="gelu")
        self.esm_transformer=nn.TransformerEncoder(encoder_layer,num_layers=2)
        self.amine_proj=nn.Sequential(nn.Linear(768,512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,256),nn.GELU())
        self.ketone_proj=nn.Sequential(nn.Linear(768,512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,256),nn.GELU())
        self.fusion_fc=nn.Sequential(nn.Linear(768,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3))
        self.fusion_residual=nn.Linear(768,256)
        self.reg_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,1))
    def forward(self,esm_emb,esm_mask,mol_amine,mol_ketone):
        esm_mask=esm_mask.bool()
        esm=self.esm_proj(esm_emb)
        padding_mask=~esm_mask
        # 避免某条序列全部为padding
        empty=~esm_mask.any(dim=1)
        if empty.any():
            esm_mask=esm_mask.clone()
            esm_mask[empty,0]=True
            padding_mask=~esm_mask
        esm=self.esm_transformer(esm,src_key_padding_mask=padding_mask)
        esm_pool=(esm*esm_mask.unsqueeze(-1).float()).sum(dim=1)/esm_mask.sum(dim=1,keepdim=True).clamp(min=1)
        amine=self.amine_proj(mol_amine)
        ketone=self.ketone_proj(mol_ketone)
        fusion=torch.cat([esm_pool,amine,ketone],dim=1)
        x=self.fusion_fc(fusion)+self.fusion_residual(fusion)
        return self.reg_head(x).squeeze(-1)

# ==================== 6. Pairwise Ranking Loss（新增核心） ====================
def pairwise_ranking_loss(pred,raw_activity,group_id,min_delta=0.03):
    pred=pred.reshape(-1)
    raw_activity=raw_activity.reshape(-1)
    group_id=group_id.reshape(-1)
    group_losses=[]
    for g in torch.unique(group_id):
        idx=torch.where(group_id==g)[0]
        if len(idx)<2: continue
        pair=torch.triu_indices(len(idx),len(idx),offset=1,device=pred.device)
        i,j=idx[pair[0]],idx[pair[1]]
        true_diff=raw_activity[i]-raw_activity[j]
        valid=true_diff.abs()>=min_delta
        if valid.any():
            sign=torch.sign(true_diff[valid])
            pred_diff=pred[i[valid]]-pred[j[valid]]
            group_losses.append(F.softplus(-sign*pred_diff).mean())
    if not group_losses: return pred.sum()*0.0
    return torch.stack(group_losses).mean()

# ==================== 7. 评估指标 ====================
def safe_corr(func,y_true,y_pred):
    if len(y_true)<2 or np.std(y_true)==0 or np.std(y_pred)==0: return np.nan
    return float(func(y_true,y_pred)[0])

def calculate_metrics(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=np.float64)
    y_pred=np.asarray(y_pred,dtype=np.float64)
    y_pred=np.maximum(y_pred,0)
    k=min(TOP_K,len(y_true))
    true_top=np.argsort(-y_true)[:k]
    pred_top=np.argsort(-y_pred)[:k]
    overlap=len(set(true_top)&set(pred_top))/max(k,1)
    ndcg=float(ndcg_score(y_true.reshape(1,-1),y_pred.reshape(1,-1),k=k)) if len(y_true)>0 and np.any(y_true>0) else 0.0
    active=y_true>=ACTIVE_THRESHOLD
    return {
        "MAE":float(mean_absolute_error(y_true,y_pred)),
        "RMSE":float(np.sqrt(mean_squared_error(y_true,y_pred))),
        "R2":float(r2_score(y_true,y_pred)) if len(y_true)>1 else np.nan,
        "Pearson":safe_corr(pearsonr,y_true,y_pred),
        "Spearman":safe_corr(spearmanr,y_true,y_pred),
        "Active_Spearman":safe_corr(spearmanr,y_true[active],y_pred[active]) if active.sum()>1 else np.nan,
        "NDCG@100":ndcg,
        "Top100_Overlap":overlap,
        "Top100_True_Mean":float(y_true[true_top].mean()) if k else np.nan,
        "Active_Count":int(active.sum())
    }

@torch.no_grad()
def evaluate(model,loader,criterion):
    model.eval()
    total_loss=0.0
    all_pred_log,all_true_log,all_true=[],[],[]
    for esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id in loader:
        esm_emb=esm_emb.to(DEVICE,non_blocking=True)
        esm_mask=esm_mask.to(DEVICE,non_blocking=True).bool()
        mol_amine=mol_amine.to(DEVICE,non_blocking=True)
        mol_ketone=mol_ketone.to(DEVICE,non_blocking=True)
        reg_label=reg_label.to(DEVICE,non_blocking=True)
        pred=model(esm_emb,esm_mask,mol_amine,mol_ketone)
        loss=criterion(pred,reg_label)
        total_loss+=loss.item()*len(reg_label)
        all_pred_log.extend(pred.detach().cpu().numpy())
        all_true_log.extend(reg_label.cpu().numpy())
        all_true.extend(raw_activity.numpy())
    pred_activity=np.expm1(np.maximum(np.asarray(all_pred_log),0))
    metrics=calculate_metrics(all_true,pred_activity)
    metrics["Loss"]=total_loss/len(loader.dataset)
    return metrics,np.asarray(all_true),pred_activity

# ==================== 8. 训练 ====================
def train_model():
    model=EnzymeSubstrateRegressor().to(DEVICE)
    criterion=nn.SmoothL1Loss(beta=0.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=10,min_lr=1e-6)
    best_metric=-float("inf")
    best_epoch=0
    wait=0
    history=[]
    for epoch in range(1,EPOCHS+1):
        model.train()
        train_loss_sum=0.0
        train_reg_sum=0.0
        train_rank_sum=0.0
        for esm_emb,esm_mask,mol_amine,mol_ketone,reg_label,raw_activity,group_id in train_loader:
            esm_emb=esm_emb.to(DEVICE,non_blocking=True)
            esm_mask=esm_mask.to(DEVICE,non_blocking=True).bool()
            mol_amine=mol_amine.to(DEVICE,non_blocking=True)
            mol_ketone=mol_ketone.to(DEVICE,non_blocking=True)
            reg_label=reg_label.to(DEVICE,non_blocking=True)
            raw_activity=raw_activity.to(DEVICE,non_blocking=True)
            group_id=group_id.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred=model(esm_emb,esm_mask,mol_amine,mol_ketone)
            reg_loss=criterion(pred,reg_label)
            rank_loss=pairwise_ranking_loss(pred,raw_activity,group_id,min_delta=RANK_MIN_DELTA)
            loss=reg_loss+RANK_WEIGHT*rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step()
            n=len(reg_label)
            train_loss_sum+=loss.item()*n
            train_reg_sum+=reg_loss.item()*n
            train_rank_sum+=rank_loss.item()*n
        train_loss=train_loss_sum/len(train_set)
        train_reg=train_reg_sum/len(train_set)
        train_rank=train_rank_sum/len(train_set)
        val_metrics,_,_=evaluate(model,val_loader,criterion)
        val_ndcg=val_metrics["NDCG@100"]
        monitor=val_ndcg if np.isfinite(val_ndcg) else -val_metrics["Loss"]
        scheduler.step(monitor)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss={train_loss:.4f} | Reg Loss={train_reg:.4f} | Rank Loss={train_rank:.4f} | Val Loss={val_metrics['Loss']:.4f} | MAE={val_metrics['MAE']:.4f} | RMSE={val_metrics['RMSE']:.4f} | R2={val_metrics['R2']:.4f} | Spearman={val_metrics['Spearman']:.4f} | Active Spearman={val_metrics['Active_Spearman']:.4f} | NDCG@100={val_ndcg:.4f} | Top100 Overlap={val_metrics['Top100_Overlap']:.4f}")
        row={"Epoch":epoch,"Train_Loss":train_loss,"Train_Reg_Loss":train_reg,"Train_Rank_Loss":train_rank,**{f"Val_{k}":v for k,v in val_metrics.items()}}
        history.append(row)
        if monitor>best_metric:
            best_metric=monitor
            best_epoch=epoch
            wait=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"best_val_ndcg":val_ndcg,"config":{"rank_weight":RANK_WEIGHT,"rank_min_delta":RANK_MIN_DELTA}},os.path.join(OUT_DIR,"best_model.pth"))
            print("保存最佳模型！")
        else:
            wait+=1
        if wait>=PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}, best NDCG@100={best_metric:.4f}")
            break
    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR,"training_history.csv"),index=False)
    checkpoint=torch.load(os.path.join(OUT_DIR,"best_model.pth"),map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics,y_true,y_pred=evaluate(model,test_loader,criterion)
    print("\n===== Best Model Test Results =====")
    print(f"Best Epoch={checkpoint['epoch']}")
    for k,v in test_metrics.items(): print(f"{k}={v}")
    pred_df=df_test.copy()
    pred_df["Activity_true"]=y_true
    pred_df["Activity_pred"]=y_pred
    pred_df["Rank_true"]=pred_df["Activity_true"].rank(method="min",ascending=False).astype(int)
    pred_df["Rank_pred"]=pred_df["Activity_pred"].rank(method="min",ascending=False).astype(int)
    pred_df.sort_values("Activity_pred",ascending=False).to_csv(os.path.join(OUT_DIR,"test_predictions.csv"),index=False)
    pred_df.nlargest(min(TOP_K,len(pred_df)),"Activity_pred").to_csv(os.path.join(OUT_DIR,"test_top100.csv"),index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(OUT_DIR,"test_metrics.csv"),index=False)
    with open(os.path.join(OUT_DIR,"config.json"),"w") as f:
        json.dump({"SEED":SEED,"BATCH_SIZE":BATCH_SIZE,"LR":LR,"WEIGHT_DECAY":WEIGHT_DECAY,"RANK_WEIGHT":RANK_WEIGHT,"RANK_MIN_DELTA":RANK_MIN_DELTA,"ACTIVE_THRESHOLD":ACTIVE_THRESHOLD,"best_epoch":int(checkpoint["epoch"])},f,indent=2)
    print(f"\n结果已保存至: {OUT_DIR}")

if __name__=="__main__":
    train_model()
