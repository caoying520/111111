import os
import random
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 1. 参数
# =========================================================
SEED=123
BATCH_SIZE=256
NUM_CLASSES=3
ACTIVITY_THRESHOLD_1=0.1
ACTIVITY_THRESHOLD_2=1.0
STAGE1_EPOCHS=50
STAGE1_LR=1e-3
STAGE2_EPOCHS=500
STAGE2_LR=3e-5
WEIGHT_DECAY=1e-5
REDUCE_FACTOR=0.5
REDUCE_PATIENCE=15
EARLY_STOP_PATIENCE=30

DATA_PATH="/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/train_data3.csv"
ESM_PATH="/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/esm_feats_emb_mask_test_6_0.pkl"
MOL_PATH="/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/SM.pkl"
SPLIT_DIR="/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/splits_Activity_high"
PRETRAINED_PATH="/data/caoying/softwares/酶R2/enzyme_交接20260609/enzyme_train/model/reg/model_resnet_krd_conc_S2_KRD_M_55%_add/best_model_add.pth"
OUTPUT_DIR="/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_finetune_3cls"
os.makedirs(OUTPUT_DIR,exist_ok=True)

def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

set_seed(SEED)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

# =========================================================
# 2. 三分类标签
# =========================================================
def activity_to_class(activity):
    activity=np.asarray(activity,dtype=float)
    return np.where(activity<ACTIVITY_THRESHOLD_1,0,np.where(activity<ACTIVITY_THRESHOLD_2,1,2)).astype(np.int64)

# =========================================================
# 3. Dataset
# =========================================================
class EnzymeSubstrateDataset(Dataset):
    def __init__(self,df,esm_dict,mol_dict):
        self.df=df.reset_index(drop=True)
        self.esm_dict=esm_dict
        self.mol_dict=mol_dict
        self.labels=activity_to_class(self.df["Activity"].values)

    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        row=self.df.iloc[i]
        name=str(row["Name"])
        sm=str(row["SM"])
        esm_item=self.esm_dict[name]
        mol_feat=self.mol_dict[sm]
        esm_feat=esm_item["emb"].float() if isinstance(esm_item,dict) else esm_item.float()
        esm_mask=esm_item["mask"].float() if isinstance(esm_item,dict) else torch.ones(esm_feat.shape[0],dtype=torch.float32)
        mol_feat=torch.as_tensor(mol_feat,dtype=torch.float32)
        sm_load=torch.tensor([float(row["SM_norm"])],dtype=torch.float32)
        enz_load=torch.tensor([float(row["Enz_norm"])],dtype=torch.float32)
        label=torch.tensor(self.labels[i],dtype=torch.long)
        return esm_feat,esm_mask,mol_feat,sm_load,enz_load,label

# =========================================================
# 4. Collate
# =========================================================
def collate_fn(batch):
    esm_feats,esm_masks,mol_feats,sm_loads,enz_loads,labels=zip(*batch)
    max_len=max(x.shape[0] for x in esm_feats)
    feat_dim=esm_feats[0].shape[1]
    esm_batch=torch.zeros(len(batch),max_len,feat_dim,dtype=torch.float32)
    mask_batch=torch.zeros(len(batch),max_len,dtype=torch.float32)
    for i,(feat,mask) in enumerate(zip(esm_feats,esm_masks)):
        L=min(feat.shape[0],max_len)
        esm_batch[i,:L]=feat[:L]
        mask_batch[i,:L]=mask[:L]
    mol_batch=torch.stack(mol_feats)
    sm_batch=torch.stack(sm_loads)
    enz_batch=torch.stack(enz_loads)
    label_batch=torch.stack(labels)
    return esm_batch,mask_batch,mol_batch,sm_batch,enz_batch,label_batch

# =========================================================
# 5. 模型
# =========================================================
class EnzymeSubstrateThreeClassClassifier(nn.Module):
    def __init__(self,dropout=0.2):
        super().__init__()
        self.esm_proj=nn.Sequential(nn.Linear(1280,512),nn.ReLU(),nn.Dropout(dropout),nn.Linear(512,256))
        self.mol_proj=nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Dropout(dropout),nn.Linear(512,256))
        encoder_layer=nn.TransformerEncoderLayer(d_model=256,nhead=8,dim_feedforward=512,dropout=dropout,batch_first=True)
        self.esm_encoder=nn.TransformerEncoder(encoder_layer,num_layers=2,enable_nested_tensor=False)
        self.sm_proj=nn.Sequential(nn.Linear(1,64),nn.ReLU(),nn.Linear(64,128),nn.LayerNorm(128))
        self.enz_proj=nn.Sequential(nn.Linear(1,64),nn.ReLU(),nn.Linear(64,128),nn.LayerNorm(128))
        input_dim=256+256+128+128
        self.fc1=nn.Linear(input_dim,256)
        self.bn1=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256,256)
        self.bn2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,128)
        self.bn3=nn.BatchNorm1d(128)
        self.dropout=nn.Dropout(0.2)
        self.classifier=nn.Linear(128,NUM_CLASSES)

    def forward(self,esm_feat,esm_mask,mol_feat,sm_load,enz_load):
        esm_feat=self.esm_proj(esm_feat)
        src_mask=(esm_mask==0).bool()
        esm_feat=self.esm_encoder(esm_feat,src_key_padding_mask=src_mask)
        mask=esm_mask.unsqueeze(-1).float()
        esm_pooled=(esm_feat*mask).sum(dim=1)/(mask.sum(dim=1).clamp(min=1e-8))
        mol_proj=self.mol_proj(mol_feat)
        sm_proj=self.sm_proj(sm_load)
        enz_proj=self.enz_proj(enz_load)
        x=torch.cat([esm_pooled,enz_proj,mol_proj,sm_proj],dim=-1)
        x=F.relu(self.bn1(self.fc1(x)))
        residual=x
        x=F.relu(self.bn2(self.fc2(x)))
        x=self.dropout(x)
        x=x+residual
        x=F.relu(self.bn3(self.fc3(x)))
        logits=self.classifier(x)
        return logits

# =========================================================
# 6. 加载原始预训练权重
# =========================================================
def load_pretrained(model,checkpoint_path):
    checkpoint=torch.load(checkpoint_path,map_location="cpu")
    if isinstance(checkpoint,dict) and "state_dict" in checkpoint: state_dict=checkpoint["state_dict"]
    elif isinstance(checkpoint,dict) and "model" in checkpoint and isinstance(checkpoint["model"],dict): state_dict=checkpoint["model"]
    else: state_dict=checkpoint
    state_dict={k.replace("module.",""):v for k,v in state_dict.items()}
    state_dict.pop("fc4.weight",None)
    state_dict.pop("fc4.bias",None)
    missing,unexpected=model.load_state_dict(state_dict,strict=False)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    print("Pretrained loaded.")
    print("Missing keys:",missing)
    print("Unexpected keys:",unexpected)
    return model

# =========================================================
# 7. 读取数据
# =========================================================
print("\nLoading data...")
df=pd.read_csv(DATA_PATH)
esm_dict=pd.read_pickle(ESM_PATH)
mol_dict=pd.read_pickle(MOL_PATH)

print("Total data:",len(df))
print("ESM:",len(esm_dict))
print("MolFormer:",len(mol_dict))

# =========================================================
# 8. 读取 split
# =========================================================
def read_split(path):
    values=pd.read_csv(path,header=None).iloc[:,0].astype(str).tolist()
    return set(values)

train_ids=read_split(os.path.join(SPLIT_DIR,"train.txt"))
val_ids=read_split(os.path.join(SPLIT_DIR,"val.txt"))
test_ids=read_split(os.path.join(SPLIT_DIR,"test.txt"))

df["Name"]=df["Name"].astype(str)
df_train=df[df["Name"].isin(train_ids)].copy()
df_val=df[df["Name"].isin(val_ids)].copy()
df_test=df[df["Name"].isin(test_ids)].copy()

print("Train:",len(df_train))
print("Val:",len(df_val))
print("Test:",len(df_test))

# =========================================================
# 9. 检查三分类分布
# =========================================================
train_cls=activity_to_class(df_train["Activity"].values)
val_cls=activity_to_class(df_val["Activity"].values)
test_cls=activity_to_class(df_test["Activity"].values)

print("\n===== Class distribution =====")
for name,y in [("Train",train_cls),("Val",val_cls),("Test",test_cls)]:
    counts=np.bincount(y,minlength=NUM_CLASSES)
    print(f"{name}: Class0={counts[0]}, Class1={counts[1]}, Class2={counts[2]}, Total={len(y)}")

# =========================================================
# 10. DataLoader
# =========================================================
train_dataset=EnzymeSubstrateDataset(df_train,esm_dict,mol_dict)
val_dataset=EnzymeSubstrateDataset(df_val,esm_dict,mol_dict)
test_dataset=EnzymeSubstrateDataset(df_test,esm_dict,mol_dict)

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True,collate_fn=collate_fn)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True,collate_fn=collate_fn)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True,collate_fn=collate_fn)

# =========================================================
# 11. 类别权重
# =========================================================
class_counts=np.bincount(train_cls,minlength=NUM_CLASSES)
class_weights=len(train_cls)/(NUM_CLASSES*np.maximum(class_counts,1))
class_weights=torch.tensor(class_weights,dtype=torch.float32,device=device)

print("\nClass counts:",class_counts)
print("Class weights:",class_weights.detach().cpu().numpy())

# =========================================================
# 12. 初始化模型
# =========================================================
model=EnzymeSubstrateThreeClassClassifier(dropout=0.2)
model=load_pretrained(model,PRETRAINED_PATH)
model=model.to(device)

criterion=nn.CrossEntropyLoss(weight=class_weights)

# =========================================================
# 13. Train Epoch
# =========================================================
def train_epoch(model,loader,criterion,optimizer,device):
    model.train()
    total_loss=0.0
    total_num=0
    for esm_feat,esm_mask,mol_feat,sm_load,enz_load,label in loader:
        esm_feat=esm_feat.to(device,non_blocking=True)
        esm_mask=esm_mask.to(device,non_blocking=True)
        mol_feat=mol_feat.to(device,non_blocking=True)
        sm_load=sm_load.to(device,non_blocking=True)
        enz_load=enz_load.to(device,non_blocking=True)
        label=label.to(device,non_blocking=True)
        optimizer.zero_grad()
        logits=model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)
        loss=criterion(logits,label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
        optimizer.step()
        total_loss+=loss.item()*label.size(0)
        total_num+=label.size(0)
    return total_loss/total_num

# =========================================================
# 14. Predict
# =========================================================
def predict(model,loader,device):
    model.eval()
    all_prob=[]
    all_label=[]
    with torch.no_grad():
        for esm_feat,esm_mask,mol_feat,sm_load,enz_load,label in loader:
            esm_feat=esm_feat.to(device,non_blocking=True)
            esm_mask=esm_mask.to(device,non_blocking=True)
            mol_feat=mol_feat.to(device,non_blocking=True)
            sm_load=sm_load.to(device,non_blocking=True)
            enz_load=enz_load.to(device,non_blocking=True)
            logits=model(esm_feat,esm_mask,mol_feat,sm_load,enz_load)
            prob=torch.softmax(logits,dim=1)
            all_prob.append(prob.cpu().numpy())
            all_label.append(label.numpy())
    probabilities=np.concatenate(all_prob,axis=0)
    labels=np.concatenate(all_label,axis=0).astype(int)
    predictions=np.argmax(probabilities,axis=1)
    return labels,probabilities,predictions

# =========================================================
# 15. 多分类评价
# =========================================================
def evaluate(y_true,y_prob,y_pred):
    accuracy=accuracy_score(y_true,y_pred)
    precision_macro=precision_score(y_true,y_pred,average="macro",zero_division=0)
    recall_macro=recall_score(y_true,y_pred,average="macro",zero_division=0)
    f1_macro=f1_score(y_true,y_pred,average="macro",zero_division=0)
    f1_weighted=f1_score(y_true,y_pred,average="weighted",zero_division=0)
    f1_each=f1_score(y_true,y_pred,average=None,labels=[0,1,2],zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    try:
        auc_macro=roc_auc_score(y_true,y_prob,multi_class="ovr",average="macro",labels=[0,1,2])
    except ValueError:
        auc_macro=np.nan
    return {"Accuracy":accuracy,"Precision_macro":precision_macro,"Recall_macro":recall_macro,"F1_macro":f1_macro,"F1_weighted":f1_weighted,"F1_class0":f1_each[0],"F1_class1":f1_each[1],"F1_class2":f1_each[2],"AUC_macro":auc_macro,"CM":cm}

# =========================================================
# 16. Stage 1：冻结特征提取部分，只训练分类头
# =========================================================
print("\n" + "="*70)
print("Stage 1: Freeze backbone, train classifier")
print("="*70)

for p in model.parameters(): p.requires_grad=False
for p in model.classifier.parameters(): p.requires_grad=True

optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=STAGE1_LR,weight_decay=WEIGHT_DECAY)

best_f1=-1
best_state=None
best_epoch=0

for epoch in range(1,STAGE1_EPOCHS+1):
    train_loss=train_epoch(model,train_loader,criterion,optimizer,device)
    y_val,prob_val,pred_val=predict(model,val_loader,device)
    metrics=evaluate(y_val,prob_val,pred_val)
    print(f"[Stage1][{epoch:03d}/{STAGE1_EPOCHS}] Loss={train_loss:.5f} Acc={metrics['Accuracy']:.4f} F1_macro={metrics['F1_macro']:.4f} F1_0={metrics['F1_class0']:.4f} F1_1={metrics['F1_class1']:.4f} F1_2={metrics['F1_class2']:.4f} AUC={metrics['AUC_macro']:.4f}")
    if metrics["F1_macro"]>best_f1:
        best_f1=metrics["F1_macro"]
        best_state=deepcopy(model.state_dict())
        best_epoch=epoch

model.load_state_dict(best_state)
print(f"Stage1 best epoch={best_epoch}, best F1_macro={best_f1:.4f}")

# =========================================================
# 17. Stage 2：解冻全部模型进行微调
# =========================================================
print("\n" + "="*70)
print("Stage 2: Fine-tune entire model")
print("="*70)

for p in model.parameters(): p.requires_grad=True

optimizer=torch.optim.AdamW(model.parameters(),lr=STAGE2_LR,weight_decay=WEIGHT_DECAY)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=REDUCE_FACTOR,patience=REDUCE_PATIENCE)

best_f1=-1
best_state=None
best_epoch=0
early_stop_count=0

best_path=os.path.join(OUTPUT_DIR,"best_model_3class.pth")

for epoch in range(1,STAGE2_EPOCHS+1):
    train_loss=train_epoch(model,train_loader,criterion,optimizer,device)
    y_val,prob_val,pred_val=predict(model,val_loader,device)
    metrics=evaluate(y_val,prob_val,pred_val)
    current_lr=optimizer.param_groups[0]["lr"]
    scheduler.step(metrics["F1_macro"])
    print(f"[Stage2][{epoch:03d}/{STAGE2_EPOCHS}] Loss={train_loss:.5f} Acc={metrics['Accuracy']:.4f} F1_macro={metrics['F1_macro']:.4f} F1_weighted={metrics['F1_weighted']:.4f} F1_0={metrics['F1_class0']:.4f} F1_1={metrics['F1_class1']:.4f} F1_2={metrics['F1_class2']:.4f} AUC={metrics['AUC_macro']:.4f} LR={current_lr:.2e}")
    if metrics["F1_macro"]>best_f1:
        best_f1=metrics["F1_macro"]
        best_state=deepcopy(model.state_dict())
        best_epoch=epoch
        early_stop_count=0
        torch.save({"model":best_state,"num_classes":NUM_CLASSES,"activity_threshold_1":ACTIVITY_THRESHOLD_1,"activity_threshold_2":ACTIVITY_THRESHOLD_2,"class_weights":class_weights.detach().cpu()},best_path)
        print(f"  --> Save best model, F1_macro={best_f1:.4f}")
    else:
        early_stop_count+=1
    if early_stop_count>=EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}")
        break

# =========================================================
# 18. 加载最佳模型
# =========================================================
checkpoint=torch.load(best_path,map_location=device)
model.load_state_dict(checkpoint["model"])
model=model.to(device)

# =========================================================
# 19. Test
# =========================================================
print("\n" + "="*70)
print("Final Test")
print("="*70)

y_test,prob_test,pred_test=predict(model,test_loader,device)
test_metrics=evaluate(y_test,prob_test,pred_test)

print(f"Accuracy       : {test_metrics['Accuracy']:.4f}")
print(f"Precision_macro: {test_metrics['Precision_macro']:.4f}")
print(f"Recall_macro   : {test_metrics['Recall_macro']:.4f}")
print(f"F1_macro       : {test_metrics['F1_macro']:.4f}")
print(f"F1_weighted    : {test_metrics['F1_weighted']:.4f}")
print(f"F1_class0      : {test_metrics['F1_class0']:.4f}")
print(f"F1_class1      : {test_metrics['F1_class1']:.4f}")
print(f"F1_class2      : {test_metrics['F1_class2']:.4f}")
print(f"AUC_macro      : {test_metrics['AUC_macro']:.4f}")
print("\nConfusion Matrix:")
print(test_metrics["CM"])

# =========================================================
# 20. 保存预测结果
# =========================================================
result=df_test.reset_index(drop=True).copy()
result["True_Class"]=activity_to_class(result["Activity"].values)
result["Pred_Class"]=pred_test
result["Prob_Class_0"]=prob_test[:,0]
result["Prob_Class_1"]=prob_test[:,1]
result["Prob_Class_2"]=prob_test[:,2]

prediction_path=os.path.join(OUTPUT_DIR,"test_predictions_3class.csv")
result.to_csv(prediction_path,index=False)

# =========================================================
# 21. 保存评价结果
# =========================================================
metrics_to_save={k:v for k,v in test_metrics.items() if k!="CM"}
metrics_to_save["Confusion_Matrix"]=test_metrics["CM"].tolist()
metrics_to_save["Class_0"]="Activity < 0.1"
metrics_to_save["Class_1"]="0.1 <= Activity < 1"
metrics_to_save["Class_2"]="Activity >= 1"
metrics_to_save["Best_Epoch"]=best_epoch

with open(os.path.join(OUTPUT_DIR,"test_metrics_3class.json"),"w",encoding="utf-8") as f:
    json.dump(metrics_to_save,f,ensure_ascii=False,indent=2)

# =========================================================
# 22. 保存分类参数
# =========================================================
config={
    "seed":SEED,
    "batch_size":BATCH_SIZE,
    "num_classes":NUM_CLASSES,
    "activity_threshold_1":ACTIVITY_THRESHOLD_1,
    "activity_threshold_2":ACTIVITY_THRESHOLD_2,
    "class_definition":{
        "0":"Activity < 0.1",
        "1":"0.1 <= Activity < 1",
        "2":"Activity >= 1"
    },
    "stage1_epochs":STAGE1_EPOCHS,
    "stage1_lr":STAGE1_LR,
    "stage2_epochs":STAGE2_EPOCHS,
    "stage2_lr":STAGE2_LR,
    "weight_decay":WEIGHT_DECAY
}

with open(os.path.join(OUTPUT_DIR,"config_3class.json"),"w",encoding="utf-8") as f:
    json.dump(config,f,ensure_ascii=False,indent=2)

print("\nDone.")
print("Best model :",best_path)
print("Predictions:",prediction_path)
print("Output dir :",OUTPUT_DIR)
