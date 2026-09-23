# ============================================================
# 10. 主程序
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # ========================================================
    # ESM / MolFormer
    # ========================================================

    with open(
        ESM_PKL,
        "rb"
    ) as f:

        esm_dict = pickle.load(f)

    with open(
        MOL_PKL,
        "rb"
    ) as f:

        mol_dict = pickle.load(f)

    print(
        f"ESM={len(esm_dict)}, "
        f"MolFormer={len(mol_dict)}"
    )

    # ========================================================
    # split
    # ========================================================

    def read_split(name):

        path = os.path.join(
            SPLIT_DIR,
            name
        )

        with open(path) as f:

            return [
                x.strip()
                for x in f
                if x.strip()
            ]

    train_idx = read_split(
        TRAIN_TXT
    )

    val_idx = read_split(
        VAL_TXT
    )

    test_idx = read_split(
        TEST_TXT
    )

    train_idx = set(
        str(x) for x in train_idx
    )

    val_idx = set(
        str(x) for x in val_idx
    )

    test_idx = set(
        str(x) for x in test_idx
    )

    print(
        f"Split IDs: "
        f"train={len(train_idx)}, "
        f"val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    # ========================================================
    # CSV
    # ========================================================

    df = pd.read_csv(
        CSV_PATH
    )

    # --------------------------------------------------------
    # 必需列检查
    # --------------------------------------------------------

    required_columns = [
        "idx",
        "Name",
        "SM",
        "Activity",
        "SM loading (g/L)",
        "Enzyme loading (g/L)"
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"CSV 缺少列: {missing_columns}"
        )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    df["idx"] = (
        df["idx"]
        .astype(str)
    )

    df["Name"] = (
        df["Name"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    df["Activity"] = pd.to_numeric(
        df["Activity"],
        errors="coerce"
    )

    raw_n = len(df)

    missing_activity = (
        df["Activity"]
        .isna()
        .sum()
    )

    df = df.dropna(
        subset=["Activity"]
    ).reset_index(
        drop=True
    )

    print(
        f"\n原始数据: {raw_n}"
    )

    print(
        f"Activity 缺失: "
        f"{missing_activity}"
    )

    print(
        f"有效 Activity: "
        f"{len(df)}"
    )

    # ========================================================
    # 根据 split 划分
    # ========================================================

    df_train = (
        df[
            df["idx"].isin(train_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_val = (
        df[
            df["idx"].isin(val_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_test = (
        df[
            df["idx"].isin(test_idx)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\n实际数据量:"
    )

    print(
        f"train={len(df_train)}, "
        f"val={len(df_val)}, "
        f"test={len(df_test)}"
    )

    # ========================================================
    # 检查 train / val / test 是否有交集
    # ========================================================

    print(
        "\nSplit overlap:"
    )

    print(
        "train ∩ val =",
        len(train_idx & val_idx)
    )

    print(
        "train ∩ test =",
        len(train_idx & test_idx)
    )

    print(
        "val ∩ test =",
        len(val_idx & test_idx)
    )

    # ========================================================
    # 二分类数据分布
    # ========================================================

    for name, d in [
        ("Train", df_train),
        ("Val", df_val),
        ("Test", df_test)
    ]:

        binary = (
            d["Activity"]
            >= ACTIVITY_THRESHOLD
        ).astype(int)

        print(
            f"\n{name} binary distribution:"
        )

        print(
            f"Negative (0): "
            f"{(binary == 0).sum()} "
            f"({(binary == 0).mean():.2%})"
        )

        print(
            f"Positive (1): "
            f"{(binary == 1).sum()} "
            f"({(binary == 1).mean():.2%})"
        )

    # ========================================================
    # 浓度处理
    # 与原模型保持一致
    # ========================================================

    SM_COL = "SM loading (g/L)"
    ENZ_COL = "Enzyme loading (g/L)"

    for d in [
        df_train,
        df_val,
        df_test
    ]:

        d[SM_COL] = pd.to_numeric(
            d[SM_COL],
            errors="coerce"
        ).fillna(0).clip(
            lower=0
        )

        d[ENZ_COL] = pd.to_numeric(
            d[ENZ_COL],
            errors="coerce"
        ).fillna(0).clip(
            lower=0
        )

        d["SM_log"] = np.log1p(
            d[SM_COL]
        )

        d["Enz_log"] = np.log1p(
            d[ENZ_COL]
        )

    # --------------------------------------------------------
    # 只使用 train 计算浓度 scaler
    # --------------------------------------------------------

    sm_mean = (
        df_train["SM_log"]
        .mean()
    )

    sm_std = (
        df_train["SM_log"]
        .std()
        + 1e-12
    )

    enz_mean = (
        df_train["Enz_log"]
        .mean()
    )

    enz_std = (
        df_train["Enz_log"]
        .std()
        + 1e-12
    )

    for d in [
        df_train,
        df_val,
        df_test
    ]:

        d["SM_norm"] = (
            d["SM_log"]
            - sm_mean
        ) / sm_std

        d["Enz_norm"] = (
            d["Enz_log"]
            - enz_mean
        ) / enz_std

    # ========================================================
    # Activity 标准化
    # 只用 train
    # ========================================================

    activity_mean = (
        df_train["Activity"]
        .mean()
    )

    activity_std = (
        df_train["Activity"]
        .std()
        + 1e-12
    )

    print(
        "\nActivity mean:",
        activity_mean
    )

    print(
        "Activity std:",
        activity_std
    )

    print(
        "\nActivity distribution:"
    )

    print(
        df["Activity"].quantile(
            [
                0,
                0.01,
                0.05,
                0.25,
                0.5,
                0.75,
                0.95,
                0.99,
                1
            ]
        )
    )

    # ========================================================
    # 保存 scaler
    # ========================================================

    scaler = {

        "activity_mean":
            float(activity_mean),

        "activity_std":
            float(activity_std),

        "activity_threshold":
            float(ACTIVITY_THRESHOLD),

        "sm_mean":
            float(sm_mean),

        "sm_std":
            float(sm_std),

        "enz_mean":
            float(enz_mean),

        "enz_std":
            float(enz_std)
    }

    pd.Series(
        scaler
    ).to_json(
        os.path.join(
            MODEL_DIR,
            "activity_scaler.json"
        )
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_set = EnzymeSubstrateDataset(
        df_train,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    val_set = EnzymeSubstrateDataset(
        df_val,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    test_set = EnzymeSubstrateDataset(
        df_test,
        esm_dict,
        mol_dict,
        activity_mean,
        activity_std
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ========================================================
    # Model
    # ========================================================

    model = EnzymeSubstrateRegressor().to(
        device
    )

    model = load_pretrained(
        model,
        PRETRAINED_MODEL,
        device
    )

    # ========================================================
    # Loss
    # ========================================================

    regression_criterion = (
        nn.SmoothL1Loss()
    )

    # --------------------------------------------------------
    # 根据 train 实际类别计算 pos_weight
    # --------------------------------------------------------

    train_positive = (
        df_train["Activity"]
        >= ACTIVITY_THRESHOLD
    ).sum()

    train_negative = (
        len(df_train)
        - train_positive
    )

    if train_positive == 0:

        raise ValueError(
            "Train 中没有 Activity >= 0.03 的正样本。"
        )

    pos_weight_value = (
        train_negative
        / train_positive
    )

    print(
        "\nBinary loss:"
    )

    print(
        "Positive:",
        train_positive
    )

    print(
        "Negative:",
        train_negative
    )

    print(
        "pos_weight:",
        pos_weight_value
    )

    classification_criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
                device=device
            )
        )
    )

    # ========================================================
    # Stage 1
    # 只训练两个 head
    # ========================================================

    print(
        "\n========== Stage 1 =========="
    )

    # --------------------------------------------------------
    # 冻结全部
    # --------------------------------------------------------

    for p in model.parameters():

        p.requires_grad = False

    # --------------------------------------------------------
    # 解冻 regression head
    # --------------------------------------------------------

    for p in model.fc4.parameters():

        p.requires_grad = True

    # --------------------------------------------------------
    # 解冻 binary head
    # --------------------------------------------------------

    for p in model.binary_head.parameters():

        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.fc4.parameters(),
                "lr":
                    STAGE1_LR
            },
            {
                "params":
                    model.binary_head.parameters(),
                "lr":
                    STAGE1_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -np.inf

    stage1_path = os.path.join(
        MODEL_DIR,
        "stage1_best.pth"
    )

    for epoch in range(
        STAGE1_EPOCHS
    ):

        loss, loss_reg, loss_cls = train_epoch(
            model,
            train_loader,
            optimizer,
            regression_criterion,
            classification_criterion,
            device,
            train_mode=False
        )

        (
            true_val,
            pred_val,
            true_binary_val,
            binary_prob_val
        ) = predict(
            model,
            val_loader,
            device,
            activity_mean,
            activity_std
        )

        m = evaluate(
            true_val,
            pred_val,
            true_binary_val,
            binary_prob_val
        )

        print(
            f"Stage1 "
            f"{epoch + 1:03d}/{STAGE1_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"Reg={loss_reg:.5f} | "
            f"Cls={loss_cls:.5f} | "
            f"R2={m['R2']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"AUC={m['AUC']:.4f}"
        )

        if m["F1"] > best_f1:

            best_f1 = m["F1"]

            torch.save(
                model.state_dict(),
                stage1_path
            )

    print(
        f"\nStage1 best F1 = "
        f"{best_f1:.4f}"
    )

    # ========================================================
    # 加载 Stage1 最优
    # ========================================================

    model.load_state_dict(
        torch.load(
            stage1_path,
            map_location=device
        )
    )

    # ========================================================
    # Stage 2
    # 全模型微调
    # ========================================================

    print(
        "\n========== Stage 2 =========="
    )

    for p in model.parameters():

        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -np.inf

    best_path = os.path.join(
        MODEL_DIR,
        "best_model.pth"
    )

    for epoch in range(
        STAGE2_EPOCHS
    ):

        loss, loss_reg, loss_cls = train_epoch(
            model,
            train_loader,
            optimizer,
            regression_criterion,
            classification_criterion,
            device,
            train_mode=True
        )

        (
            true_val,
            pred_val,
            true_binary_val,
            binary_prob_val
        ) = predict(
            model,
            val_loader,
            device,
            activity_mean,
            activity_std
        )

        m = evaluate(
            true_val,
            pred_val,
            true_binary_val,
            binary_prob_val
        )

        print(
            f"Stage2 "
            f"{epoch + 1:03d}/{STAGE2_EPOCHS} | "
            f"Loss={loss:.5f} | "
            f"Reg={loss_reg:.5f} | "
            f"Cls={loss_cls:.5f} | "
            f"RMSE={m['RMSE']:.5f} | "
            f"MAE={m['MAE']:.5f} | "
            f"R2={m['R2']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"AUC={m['AUC']:.4f}"
        )

        # ----------------------------------------------------
        # 按 binary head 的 F1 保存
        # ----------------------------------------------------

        if m["F1"] > best_f1:

            best_f1 = m["F1"]

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                f"  >>> Best model: "
                f"F1={best_f1:.4f}"
            )

    # ========================================================
    # Test
    # ========================================================

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device
        )
    )

    (
        true_test,
        pred_test,
        true_binary_test,
        binary_prob_test
    ) = predict(
        model,
        test_loader,
        device,
        activity_mean,
        activity_std
    )

    test_m = evaluate(
        true_test,
        pred_test,
        true_binary_test,
        binary_prob_test
    )

    # ========================================================
    # Final test
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    for k, v in test_m.items():

        print(
            f"{k}: {v}"
        )

    # ========================================================
    # 保存预测
    # ========================================================

    result = df_test.copy()

    # regression
    result["True_Activity"] = (
        true_test
    )

    result["Pred_Activity"] = (
        pred_test
    )

    # binary ground truth
    result["True_Binary"] = (
        true_test
        >= ACTIVITY_THRESHOLD
    ).astype(int)

    # binary head probability
    result["Pred_Active_Probability"] = (
        binary_prob_test
    )

    # binary head prediction
    result["Pred_Binary"] = (
        binary_prob_test >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # 额外保存 regression head 按 0.03 阈值的结果
    # 用于比较两个方法
    # --------------------------------------------------------

    result["Pred_Binary_from_Regression"] = (
        result["Pred_Activity"]
        >= ACTIVITY_THRESHOLD
    ).astype(int)

    output_csv = os.path.join(
        MODEL_DIR,
        "test_predictions.csv"
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n预测结果已保存：",
        output_csv
    )


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    main()
