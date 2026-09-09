
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold  #同源性屏蔽与数据划分 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

from Model import ProtSATT

def parse_args():
    parser = argparse.ArgumentParser(description="ProtSATT")
    parser.add_argument('--num_classes', type=int, default=3, choices=[2, 3])
    parser.add_argument('--sim_threshold', type=str, default='Sim_40')   #它读取 Sim_40（相似度40%聚类）列。同一个聚类组的所有序列，要么全在训练集，要么全在测试集，绝不交叉
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau') # ReduceLROnPlateau LambdaLR
    parser.add_argument('--epochs', type=int, default=2500)  #原始数据默认500
    parser.add_argument('--warmup', type=int, default=150) # LambdaLR
    parser.add_argument('--patience', type=int, default=100)  #早停机制，epoch=500默认设置patience=500
    parser.add_argument('--lr_patience', type=int, default=20) # 5 ReduceLROnPlateau
    parser.add_argument('--lr_factor', type=float, default=0.5) # 0.5 ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-6) # 1e-6 ReduceLROnPlateau
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4) # 论文Table 4: E. coli = 0.0001
    parser.add_argument('--seed', type=int, default=2025) # 42
    # parser.add_argument('--data_dir', type=str, default='./datasets/EColi/E_Coli_3labels_seqID')
    # parser.add_argument('--save_dir', type=str, default=r'D:\Project\Python\conclusion\EColi\2_3_lables_experiment\checkpoints_2labels_lr1e-2_LambdaLR_monitorACC')
    parser.add_argument('--data_dir', type=str, default='/data/caoying/softwares/ProtSATT/new_train/feature')
    parser.add_argument('--save_dir', type=str, default=r'/data/caoying/softwares/ProtSATT/new_train/result/3_lables_experiment_2500/checkpoints_2labels_lr1e-2_LambdaLR_monitorACC')

    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--first_self_query_dim', type=int, default=32)
    parser.add_argument('--first_self_return_dim', type=int, default=512)
    parser.add_argument('--first_self_num_head', type=int, default=1)
    parser.add_argument('--first_self_dropout', type=int, default=0.15)
    parser.add_argument('--first_self_residual_coef', type=float, default=None) # 论文Table 4: FF residual rate, None=自动(2类→0.1, 3类→0.0)

    parser.add_argument('--self_deep', type=int, default=1)
    parser.add_argument('--deep_self_query_dim', type=int, default=16)
    parser.add_argument('--deep_self_return_dim', type=int, default=128)
    parser.add_argument('--deep_self_num_head', type=int, default=1)
    parser.add_argument('--deep_self_dropout', type=float, default=0.15)
    parser.add_argument('--deep_self_residual_coef', type=float, default=0.5) #DFE

    parser.add_argument('--deep_cross_query_dim', type=int, default=8)
    parser.add_argument('--deep_cross_return_dim', type=int, default=32)
    parser.add_argument('--deep_cross_num_head', type=int, default=1)
    parser.add_argument('--deep_cross_dropout', type=int, default=0.15)
    parser.add_argument('--deep_cross_residual_coef', type=float, default=0) #FF

    parser.add_argument('--out_scores', type=int, default=None) # None=自动跟随 num_classes (2类→2, 3类→3)
    # 独立测试集 ID 文件 (CSV, 单列, 无表头)
    # 若提供，训练时会排除这些 ID 后做 10 折 CV，每折最佳模型还会在该独立测试集上评估
    # 若不提供 (默认空字符串)，走原始 10 折 CV 流程
    parser.add_argument('--test_ids_file', type=str, default='/data/caoying/softwares/ProtSATT/new_train/feature/test_ids.csv')
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiFeatureDataset(Dataset):
    def __init__(self, x_esm, x_unirep, x_prott5, labels):
        self.x_esm = torch.tensor(x_esm, dtype=torch.float32)
        self.x_unirep = torch.tensor(x_unirep, dtype=torch.float32)
        self.x_prott5 = torch.tensor(x_prott5, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x_esm[idx], self.x_unirep[idx], self.x_prott5[idx], self.labels[idx]

def load_and_align_data(args):
    print(f"📦 loading {args.num_classes} dataset...")

    # cluster_csv_path = os.path.join(args.data_dir, 'EColi_homology_clusters_unified.csv')
    cluster_csv_path = os.path.join(args.data_dir, '5ml_homology_clusters_unified.csv')
    cluster_df = pd.read_csv(cluster_csv_path)

    #从聚类文件中提取_的最后一位作为标签（本身原始数据标签就拼接在fasta标题一起）
    cluster_df = cluster_df.dropna(subset=['Sequence_ID']).copy()
    def safe_extract_label(seq_id):
        try:
            return int(str(seq_id).split('_')[-1])   #它通过下划线切分，取最后一部分并转为整数
        except ValueError:
            return -1
    cluster_df['Label'] = cluster_df['Sequence_ID'].apply(safe_extract_label)  #这一行把提取出来的数字作为模型的Label，使用标签文件中的 Sequence_ID列进行映射
    if (cluster_df['Label'] == -1).sum() > 0:
        cluster_df = cluster_df[cluster_df['Label'] != -1].copy()

    if args.num_classes == 2:   #二分类会将标签 0 视为“不可溶”，标签 2 视为“可溶”（代码中 isin([0, 2]) 逻辑）。
        df_filtered = cluster_df[cluster_df['Label'].isin([0, 2])].copy()     #目前代码只取了 Label 为 0 和 2 的数据
        df_filtered['Label'] = df_filtered['Label'].replace({2: 1})
    else:
        df_filtered = cluster_df.copy()  #三分类: 不过滤，全部保留，labels: 0, 1, 2 原样不动

    print(f"-> filtered remaining: {len(df_filtered)}")

   
    # #从id文件中提取标签（我的krd数据处理的标签是放在新文件中，而不是拼接在一起，读取文件后再直接贴在了聚类索引表的右侧，变成了一个新的列True_Label）
    #  # 2. 读取纯标签文件 (假设文件名是 y_labels.csv，且只有一列数字)
    # label_csv_path = os.path.join(args.data_dir, 'y.csv')
    # # header=None 表示没有表头，第一行就是数据
    # label_df = pd.read_csv(label_csv_path, header=None)
    # # 检查行数是否绝对一致
    # if len(cluster_df) != len(label_df):
    #     raise ValueError(f"❌ 行数不匹配！聚类文件有 {len(cluster_df)} 行，但标签文件有 {len(label_df)} 行。")

    # # 3. 直接按行号把标签贴上去
    # # 取 label_df 的第一列 (index 0)
    # cluster_df['True_Label'] = label_df.iloc[:, 0].values  #将标签加载到聚类文件中作为一列

    # # 4. 定义你的 KRD 映射逻辑 (根据你之前的 -1, 0, >0 需求)
    # def map_krd_labels(val):
    #     val = int(val)
    #     if args.num_classes == 3:
    #         if val <= 0: return 0  # 设置标签小于0的数对应0，表示不溶解
    #         if 0 < val <= 5:  return 1  # 低溶解
    #         if val > 5:   return 2  # 设置标签大于5的数对应2，表示高溶解且有表达
    #     elif args.num_classes == 2:
    #         # 如果是二分类，只看 成功 vs 失败
    #         return 1 if val > 0 else 0
    #     return 0

    # # 使用标签文件中的 True_Label 列进行映射
    # cluster_df['Label'] = cluster_df['True_Label'].apply(map_krd_labels)
    # # 打印分布情况，帮你确认数据是否正确映射
    # counts = cluster_df['Label'].value_counts().to_dict()
    # print(f"📊 映射后分布 -> Class 0(不溶/不表达): {counts.get(0,0)}, Class 1(低溶): {counts.get(1,0)}, Class 2(高溶): {counts.get(2,0)}")
    # # 后续的过滤和对齐特征逻辑保持不变...
    # # (注意：df_filtered 现在会使用合并后的 cluster_df)
    # df_filtered = cluster_df.copy()



    groups = df_filtered[args.sim_threshold].values     #它从 Sim_40 列里取出了分组信息（用于交叉验证）
    labels = df_filtered['Label'].values
    target_seq_ids = df_filtered['Sequence_ID'].values

    print("⏳ aligning Sequence_ID...")

    def load_and_match_features(csv_filename, target_ids):
        filepath = os.path.join(args.data_dir, csv_filename)
        df_feat = pd.read_csv(filepath, header=None)

        df_feat.rename(columns={0: 'Sequence_ID'}, inplace=True)
        df_feat.set_index('Sequence_ID', inplace=True)

        try:
            aligned_df = df_feat.loc[target_ids]
        except KeyError as e:
            raise KeyError(f"❌ Error： {csv_filename} can't find sequence ID。\nmsg: {e}")

        return aligned_df.values

    X_esm = load_and_match_features('x_5ml_esm2_dataset_with_IDs.csv', target_seq_ids)
    X_unirep = load_and_match_features('x_5ml_unirep_dataset_with_IDs.csv', target_seq_ids)
    X_prott5 = load_and_match_features('x_5ml_protT5_dataset_with_IDs.csv', target_seq_ids)

    print(f"✅ feature ID aligned - ESM2: {X_esm.shape}, UniRep: {X_unirep.shape}, ProtT5: {X_prott5.shape}")

    # ===== 独立测试集分离 =====
    # 若提供 test_ids_file，则把对应 ID 的样本从训练数据中剔除，作为独立测试集返回
    # 这样可以保证微调/调参时测试集始终固定不变
    test_data = None
    if args.test_ids_file:
        test_ids_set = set(pd.read_csv(args.test_ids_file, header=None)[0].astype(str))
        is_test = np.array([str(sid) in test_ids_set for sid in target_seq_ids])
        n_test = int(is_test.sum())
        if n_test == 0:
            print(f"⚠️ --test_ids_file 中没有任何 ID 匹配到当前数据集，将忽略独立测试集划分")
        else:
            train_mask = ~is_test
            test_mask = is_test
            # 切出独立测试集
            test_data = {
                'X_esm':    X_esm[test_mask],
                'X_unirep': X_unirep[test_mask],
                'X_prott5': X_prott5[test_mask],
                'labels':   labels[test_mask],
                'groups':   groups[test_mask],
                'seq_ids':  target_seq_ids[test_mask],
            }
            # 训练数据 = 排除独立测试集后的剩余样本
            X_esm, X_unirep, X_prott5 = X_esm[train_mask], X_unirep[train_mask], X_prott5[train_mask]
            labels, groups, target_seq_ids = labels[train_mask], groups[train_mask], target_seq_ids[train_mask]
            print(f"📦 独立测试集: {n_test} 样本 | 训练数据: {int(train_mask.sum())} 样本")

    return X_esm, X_unirep, X_prott5, labels, groups, target_seq_ids, test_data

def evaluate(model, dataloader, criterion, device, num_classes, args):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_esm, x_unirep, x_prott5, y in dataloader:
            x_esm, x_unirep, x_prott5, y = x_esm.to(device), x_unirep.to(device), x_prott5.to(device), y.to(device)

            # logits = model(x_esm, x_unirep, x_prott5)

            logits = model(x_esm, x_unirep, x_prott5, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=True,
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            probs = torch.softmax(logits, dim=1)  #三分类就是三个 0 到 1 之间的小数，且相加等于 1
            preds = torch.argmax(probs, dim=1)     #返回的是最大值所在的下标（索引），最终类别，输出整数0，1，2，二分类就输出0，1

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_loss /= len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics = {
        'Loss': total_loss,
        'Accuracy': acc,
        'MCC': mcc
    }

    if num_classes == 2:
        metrics['Precision'] = precision_score(all_labels, all_preds, zero_division=0)
        metrics['Recall'] = recall_score(all_labels, all_preds, zero_division=0)
        metrics['F1'] = f1_score(all_labels, all_preds, zero_division=0)

        pos_probs = [p[1] for p in all_probs]
        metrics['ROC_AUC'] = roc_auc_score(all_labels, pos_probs)
        metrics['PR_AUC'] = average_precision_score(all_labels, pos_probs)

        monitor_auc = metrics['ROC_AUC']
    else:
        metrics['Macro_Precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['Macro_Recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['Macro_F1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics['Macro_ROC_AUC'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        monitor_auc = metrics['Macro_ROC_AUC']

    return monitor_auc, metrics, all_labels, all_preds, all_probs

def main():
    args = parse_args()
    # 论文 Table 4: FF residual rate 按分类数自动设置 (2类→0.1, 3类→0.0)
    if args.first_self_residual_coef is None:
        args.first_self_residual_coef = 0.1 if args.num_classes == 2 else 0.0
    # out_scores 自动跟随 num_classes，避免标签越界
    if args.out_scores is None:
        args.out_scores = args.num_classes
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    run_name = f"ProtSATT_{args.num_classes}Class_{args.sim_threshold}"
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"=== Starting ProtSATT {args.num_classes}-Class Training on {device} ===")
    X_esm, X_unirep, X_prott5, labels, groups, seq_ids, test_data = load_and_align_data(args)

    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    outer_fold_results = []
    independent_test_results = []  # 独立测试集 (若提供) 每折最佳模型的评估结果

    for fold, (train_val_idx, test_idx) in enumerate(sgkf_outer.split(X_esm, labels, groups)):
        print(f"\n" + "="*50)
        print(f"🚀 Outer Fold {fold+1}/10")

        X_tv_esm, X_tv_uni, X_tv_pro = X_esm[train_val_idx], X_unirep[train_val_idx], X_prott5[train_val_idx]
        y_tv, groups_tv = labels[train_val_idx], groups[train_val_idx]

        sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + fold)
        inner_train_sub_idx, inner_val_sub_idx = next(sgkf_inner.split(X_tv_esm, y_tv, groups_tv))

        train_idx = train_val_idx[inner_train_sub_idx]
        val_idx = train_val_idx[inner_val_sub_idx]

        train_groups, val_groups, test_groups = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
        assert train_groups.isdisjoint(val_groups) and train_groups.isdisjoint(test_groups) and val_groups.isdisjoint(test_groups), "⚠️ 严重数据泄露！"

        train_dataset = MultiFeatureDataset(X_esm[train_idx], X_unirep[train_idx], X_prott5[train_idx], labels[train_idx])
        val_dataset = MultiFeatureDataset(X_esm[val_idx], X_unirep[val_idx], X_prott5[val_idx], labels[val_idx])
        test_dataset = MultiFeatureDataset(X_esm[test_idx], X_unirep[test_idx], X_prott5[test_idx], labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # model = ProtSATT(num_classes=args.num_classes).to(device)
        model = ProtSATT(
            dropout=args.dropout,
            first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep,
            deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
            out_scores=args.out_scores,
        ).to(device)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.03)

        criterion = nn.CrossEntropyLoss()

        scheduler = None
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
            )
        elif args.scheduler == 'LambdaLR':
            def lambda_lr(s):
                warm_up = args.warmup
                s += 1
                return (32 ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0

        model_save_path = os.path.join(save_path, f"best_model_fold_{fold+1}.pth")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for x_esm_b, x_unirep_b, x_prott5_b, y_b in train_loader:
                x_esm_b, x_unirep_b, x_prott5_b, y_b = x_esm_b.to(device), x_unirep_b.to(device), x_prott5_b.to(device), y_b.to(device)

                optimizer.zero_grad()
                # logits = model(x_esm_b, x_unirep_b, x_prott5_b)
                logits = model(x_esm_b, x_unirep_b, x_prott5_b, device=device,
                                     first_self_query_dim=args.first_self_query_dim,
                                     deep_self=True,
                                     deep_self_query_dim=args.deep_self_query_dim,
                                     deep_cross_query_dim=args.deep_cross_query_dim)

                loss = criterion(logits, y_b)
                total_loss += loss.item() * y_b.size(0)
                loss.backward()
                optimizer.step()
                if args.scheduler == 'LambdaLR':
                    scheduler.step()
            print(f"   Train Epoch {epoch+1:03d} [] | Train Loss: {total_loss / len(train_loader.dataset)}")
            # 验证评估 (利用字典解包避免混乱)
            val_monitor_auc, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, args.num_classes, args)

            if args.scheduler == 'ReduceLROnPlateau':
                # scheduler.step(val_monitor_auc)
                scheduler.step(val_metrics['Accuracy'])

            current_lr = optimizer.param_groups[0]['lr']

            if val_metrics['Accuracy'] > best_val_auc:
                best_val_auc = val_metrics['Accuracy']
            # if val_monitor_auc > best_val_auc:
            #     best_val_auc = val_monitor_auc
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                patience_counter = 0
                print(f"   Val Epoch {epoch+1:03d} [*] | Val ACC: {val_metrics['Accuracy']:.4f} | Val AUC: {val_monitor_auc:.4f} | Val MCC: {val_metrics['MCC']:.4f} | LR: {current_lr:.2e} | LOSS: {val_metrics['Loss']:.4f}")
            else:
                patience_counter += 1
                print(f"   Val Epoch {epoch+1:03d} [ ] | Val ACC: {val_metrics['Accuracy']:.4f} | Val AUC: {val_monitor_auc:.4f} | Val MCC: {val_metrics['MCC']:.4f} | LR: {current_lr:.2e} | Patience: {patience_counter}/{args.patience} | LOSS: {val_metrics['Loss']:.4f}")

            if patience_counter >= args.patience:
                print(f"   🛑 early patience triggered！{args.patience} epoch didn't improve。")
                break

        print(f"   ✨ Inner-Val 最佳点: Epoch {best_epoch+1} (AUC = {best_val_auc:.4f})")

        model.load_state_dict(torch.load(model_save_path))
        test_monitor_auc, test_metrics, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device, args.num_classes, args)

        if args.num_classes == 2:
            print(f"   🏆 Outer-Test 成绩 -> ROC_AUC: {test_metrics['ROC_AUC']:.4f} | PR_AUC: {test_metrics['PR_AUC']:.4f} | MCC: {test_metrics['MCC']:.4f} | F1: {test_metrics['F1']:.4f} | ACC: {test_metrics['Accuracy']:.4f}")
        else:
            print(f"   🏆 Outer-Test 成绩 -> Macro_AUC: {test_metrics['Macro_ROC_AUC']:.4f} | MCC: {test_metrics['MCC']:.4f} | Macro_F1: {test_metrics['Macro_F1']:.4f} | ACC: {test_metrics['Accuracy']:.4f}")

        test_metrics['Fold'] = fold + 1
        outer_fold_results.append(test_metrics)

        # 保存外层测试集预测结果 (每折一份)
        test_df = pd.DataFrame({
            'Sequence_ID': seq_ids[test_idx],
            'Group': groups[test_idx],
            'True_Label': test_labels,
            'Pred_Label': test_preds,
        })
        for c in range(args.num_classes):
            test_df[f'Prob_Class_{c}'] = np.array(test_probs)[:, c]
        test_df['Fold'] = fold + 1
        test_pred_path = os.path.join(save_path, f"test_predictions_fold_{fold+1}.csv")
        test_df.to_csv(test_pred_path, index=False)
        print(f"   💾 外层测试集保存至: {test_pred_path}")

        # ===== 独立测试集评估 (若提供) =====
        # 用每折最佳模型在固定独立测试集上推理，得到 10 次评估结果
        # 这样可以观察模型在不同折训练后对该固定测试集的稳定性
        if test_data is not None:
            ind_test_dataset = MultiFeatureDataset(
                test_data['X_esm'], test_data['X_unirep'], test_data['X_prott5'], test_data['labels']
            )
            ind_test_loader = DataLoader(ind_test_dataset, batch_size=args.batch_size, shuffle=False)
            ind_auc, ind_metrics, ind_labels, ind_preds, ind_probs = evaluate(
                model, ind_test_loader, criterion, device, args.num_classes, args
            )
            if args.num_classes == 2:
                print(f"   🎯 独立测试集 -> ROC_AUC: {ind_metrics['ROC_AUC']:.4f} | PR_AUC: {ind_metrics['PR_AUC']:.4f} | MCC: {ind_metrics['MCC']:.4f} | F1: {ind_metrics['F1']:.4f} | ACC: {ind_metrics['Accuracy']:.4f}")
            else:
                print(f"   🎯 独立测试集 -> Macro_AUC: {ind_metrics['Macro_ROC_AUC']:.4f} | MCC: {ind_metrics['MCC']:.4f} | Macro_F1: {ind_metrics['Macro_F1']:.4f} | ACC: {ind_metrics['Accuracy']:.4f}")

            # 保存独立测试集预测结果 (每折一份, 文件名带 independent 前缀)
            ind_df = pd.DataFrame({
                'Sequence_ID': test_data['seq_ids'],
                'Group': test_data['groups'],
                'True_Label': ind_labels,
                'Pred_Label': ind_preds,
            })
            for c in range(args.num_classes):
                ind_df[f'Prob_Class_{c}'] = np.array(ind_probs)[:, c]
            ind_df['Fold'] = fold + 1
            ind_pred_path = os.path.join(save_path, f"independent_test_fold_{fold+1}.csv")
            ind_df.to_csv(ind_pred_path, index=False)

            ind_metrics['Fold'] = fold + 1
            independent_test_results.append(ind_metrics)

    df_res = pd.DataFrame(outer_fold_results)

    cols = ['Fold'] + [c for c in df_res.columns if c != 'Fold']
    df_res = df_res[cols]

    mean_res = df_res.drop(columns=['Fold']).mean()
    std_res = df_res.drop(columns=['Fold']).std()

    report_path = os.path.join(save_path, "Rigorous_10Fold_Summary.txt")
    with open(report_path, 'w') as f:
        f.write(f"========== ProtSATT {args.num_classes}-Class Performance ({args.sim_threshold}) ==========\n\n")

        f.write("--- Fold Details ---\n")
        f.write(df_res.to_string(index=False) + "\n\n")

        f.write("--- Overall Average ---\n")
        for metric in mean_res.index:
            line = f"{metric:<15}: {mean_res[metric]:.4f} ± {std_res[metric]:.4f}\n"
            print(line.strip())
            f.write(line)

    csv_path = os.path.join(save_path, "Rigorous_10Fold_Summary.csv")
    df_res.to_csv(csv_path, index=False)

    # ===== 独立测试集汇总 (若提供) =====
    # 把 10 折模型对该固定测试集的预测结果合并、指标平均
    if test_data is not None and independent_test_results:
        df_ind = pd.DataFrame(independent_test_results)
        cols_ind = ['Fold'] + [c for c in df_ind.columns if c != 'Fold']
        df_ind = df_ind[cols_ind]

        ind_mean = df_ind.drop(columns=['Fold']).mean()
        ind_std = df_ind.drop(columns=['Fold']).std()

        ind_csv_path = os.path.join(save_path, "Independent_Test_Summary.csv")
        df_ind.to_csv(ind_csv_path, index=False)

        # 追加写入主报告
        with open(report_path, 'a') as f:
            f.write("\n\n========== 独立测试集 (固定不变) ==========\n")
            f.write(f"测试集样本数: {len(test_data['seq_ids'])}\n")
            f.write(f"测试集 ID 文件: {args.test_ids_file}\n\n")
            f.write("--- 各折模型在独立测试集上的表现 ---\n")
            f.write(df_ind.to_string(index=False) + "\n\n")
            f.write("--- 独立测试集平均 (10 折模型) ---\n")
            for metric in ind_mean.index:
                line = f"{metric:<15}: {ind_mean[metric]:.4f} ± {ind_std[metric]:.4f}\n"
                print(line.strip())
                f.write(line)

        print(f"📦 独立测试集汇总已保存: {ind_csv_path}")

    print(f"\n✅ saved: {save_path}")

if __name__ == "__main__":
    main()
