from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import pearsonr, spearmanr


def evaluate(model, loader, device, criterion=None):

    model.eval()

    all_cls, all_reg = [], []
    all_cls_y, all_reg_y = [], []
    all_raw_y = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            (esm_feat, esm_mask, mol_feat1, mol_feat2,
             cls_y, reg_y, reg_mask, raw_y) = batch

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)

            cls_y = cls_y.to(device)
            reg_y = reg_y.to(device)
            reg_mask = reg_mask.to(device)

            cls_logits, reg_output = model(
                esm_feat, esm_mask, mol_feat1, mol_feat2
            )

            if criterion is not None:
                loss, _, _ = criterion(
                    cls_logits, reg_output,
                    cls_y, reg_y, reg_mask
                )

                total_loss += loss.item() * len(raw_y)
                total_samples += len(raw_y)

            cls_prob = torch.sigmoid(cls_logits)

            # log1p(Activity) 转换回原始活性尺度
            reg_pred = torch.expm1(
                reg_output.clamp(min=0, max=10)
            )

            # 期望活性估计
            expected_activity = cls_prob * reg_pred

            all_cls.extend(cls_prob.cpu().numpy())
            all_reg.extend(reg_pred.cpu().numpy())

            all_cls_y.extend(cls_y.cpu().numpy())
            all_reg_y.extend(reg_y.cpu().numpy())
            all_raw_y.extend(raw_y.cpu().numpy())

    all_cls = np.array(all_cls)
    all_reg = np.array(all_reg)
    all_raw_y = np.array(all_raw_y)

    all_cls_y = np.array(all_cls_y).astype(int)

    expected_activity = all_cls * all_reg

    # 分类指标
    pred_class = (all_cls >= 0.5).astype(int)

    acc = accuracy_score(all_cls_y, pred_class)
    precision = precision_score(
        all_cls_y, pred_class, zero_division=0
    )
    recall = recall_score(
        all_cls_y, pred_class, zero_division=0
    )
    f1 = f1_score(
        all_cls_y, pred_class, zero_division=0
    )

    if len(np.unique(all_cls_y)) == 2:
        auc = roc_auc_score(all_cls_y, all_cls)
        pr_auc = average_precision_score(all_cls_y, all_cls)
    else:
        auc, pr_auc = np.nan, np.nan

    # 原始活性尺度下的整体预测
    mse = mean_squared_error(
        all_raw_y, expected_activity
    )
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        all_raw_y, expected_activity
    )
    r2 = r2_score(all_raw_y, expected_activity)

    # 只评价真实有效活性样本的条件回归能力
    active_mask = all_raw_y >= 0.03

    if active_mask.sum() > 1:

        active_mae = mean_absolute_error(
            all_raw_y[active_mask],
            all_reg[active_mask]
        )

        active_rmse = np.sqrt(
            mean_squared_error(
                all_raw_y[active_mask],
                all_reg[active_mask]
            )
        )

        active_r2 = r2_score(
            all_raw_y[active_mask],
            all_reg[active_mask]
        )

        active_spearman = spearmanr(
            all_raw_y[active_mask],
            all_reg[active_mask]
        ).statistic

    else:
        active_mae = np.nan
        active_rmse = np.nan
        active_r2 = np.nan
        active_spearman = np.nan

    avg_loss = (
        total_loss / total_samples
        if criterion is not None else None
    )

    metrics = {
        "Loss": avg_loss,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Active_MAE": active_mae,
        "Active_RMSE": active_rmse,
        "Active_R2": active_r2,
        "Active_Spearman": active_spearman
    }

    predictions = {
        "true": all_raw_y,
        "prob": all_cls,
        "conditional_activity": all_reg,
        "expected_activity": expected_activity
    }

    return metrics, predictions

def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    device,
    epochs=300,
    patience=30
):

    model_dir = "model_activity_multitask"
    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100
        )

        for batch in loop:

            (esm_feat, esm_mask, mol_feat1, mol_feat2,
             cls_y, reg_y, reg_mask, raw_y) = batch

            esm_feat = esm_feat.to(device)
            esm_mask = esm_mask.to(device)
            mol_feat1 = mol_feat1.to(device)
            mol_feat2 = mol_feat2.to(device)

            cls_y = cls_y.to(device)
            reg_y = reg_y.to(device)
            reg_mask = reg_mask.to(device)

            cls_logits, reg_output = model(
                esm_feat, esm_mask, mol_feat1, mol_feat2
            )

            loss, loss_cls, loss_reg = criterion(
                cls_logits,
                reg_output,
                cls_y,
                reg_y,
                reg_mask
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_reg += loss_reg.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
                reg=f"{loss_reg.item():.4f}"
            )

        train_loss = total_loss / len(train_loader)

        # 验证
        val_metrics, val_pred = evaluate(
            model, val_loader, device, criterion
        )

        val_loss = val_metrics["Loss"]

        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1={val_metrics['F1']:.4f} | "
            f"Val PR-AUC={val_metrics['PR_AUC']:.4f} | "
            f"Val MAE={val_metrics['MAE']:.4f} | "
            f"Active MAE={val_metrics['Active_MAE']:.4f} | "
            f"Active R2={val_metrics['Active_R2']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{
                "val_" + k: v
                for k, v in val_metrics.items()
            }
        })

        # 保存最佳模型
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "best_model.pth")
            )

            print("保存最佳模型！")

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"Early Stopping at Epoch {epoch+1}"
            )
            break

    pd.DataFrame(history).to_csv(
        os.path.join(model_dir, "training_history.csv"),
        index=False
    )

    print(
        f"\n训练结束，最佳验证损失：{best_val_loss:.6f}"
    )

# ========================
# Dataset
# ========================

train_set = EnzymeSubstrateDataset(
    df_train, esm_dict, mol_dict, threshold=0.03
)

val_set = EnzymeSubstrateDataset(
    df_val, esm_dict, mol_dict, threshold=0.03
)

test_set = EnzymeSubstrateDataset(
    df_test, esm_dict, mol_dict, threshold=0.03
)


# ========================
# DataLoader
# ========================

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ========================
# Model
# ========================

model = EnzymeSubstrateRegressor(
    esm_dim=1280,
    mol_dim=768,
    dropout=0.25,
    hidden_dim=256
).to(device)


# ========================
# Loss
# ========================

n_pos = (df_train["Activity"] >= 0.03).sum()
n_neg = len(df_train) - n_pos

pos_weight = n_neg / max(n_pos, 1)

criterion = MultiTaskLoss(
    pos_weight=pos_weight,
    lambda_cls=1.0,
    lambda_reg=1.0
).to(device)

print("Positive weight:", pos_weight)


# ========================
# Optimizer
# ========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ========================
# Training
# ========================

train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    device,
    epochs=300,
    patience=30
)


# ========================
# Test
# ========================

best_model_path = (
    "model_activity_multitask/best_model.pth"
)

model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)

model.to(device)

test_metrics, test_predictions = evaluate(
    model,
    test_loader,
    device,
    criterion
)

print("\n========== TEST RESULTS ==========")

for key, value in test_metrics.items():
    print(f"{key}: {value}")


# ========================
# Save predictions
# ========================

result_df = df_test.reset_index(drop=True).copy()

result_df["True_Activity"] = test_predictions["true"]

result_df["Pred_Probability"] = test_predictions["prob"]

result_df["Pred_Conditional_Activity"] = (
    test_predictions["conditional_activity"]
)

result_df["Pred_Expected_Activity"] = (
    test_predictions["expected_activity"]
)

result_df.to_csv(
    "model_activity_multitask/test_predictions.csv",
    index=False
)

print("测试集预测结果已保存！")
