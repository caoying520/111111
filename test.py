weights = len(train_idx) / (args.num_classes * np.maximum(counts, 1))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr
        )

        best_f1 = -1.0
        best_epoch = 0
        wait = 0
        ckpt_path = os.path.join(fold_dir, "best_model.pth")

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for xe, xu, xp, y in train_loader:
                xe, xu, xp, y = xe.to(device), xu.to(device), xp.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xe, xu, xp)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * y.size(0)

            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics["Macro_F1"])
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold} | Epoch {epoch:03d} | "
                f"TrainLoss {running_loss / len(train_loader.dataset):.4f} | "
                f"ValLoss {val_metrics['Loss']:.4f} | "
                f"ValF1 {val_metrics['Macro_F1']:.4f} | "
                f"ValAUC {val_metrics['Macro_ROC_AUC']:.4f} | "
                f"ValMCC {val_metrics['MCC']:.4f} | LR {lr_now:.2e}"
            )

            if val_metrics["Macro_F1"] > best_f1:
                best_f1 = val_metrics["Macro_F1"]
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                        "esm_dim": X_esm.shape[1],
                        "unirep_dim": X_uni.shape[1],
                        "prott5_dim": X_pro.shape[1],
                    },
                    ckpt_path,
                )
            else:
                wait += 1

            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val F1={best_f1:.4f}")
                break

        # 测试集只在该折模型选择完成后评估一次
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"Fold {fold} TEST | F1={test_metrics['Macro_F1']:.4f} | "
            f"AUC={test_metrics['Macro_ROC_AUC']:.4f} | "
            f"MCC={test_metrics['MCC']:.4f} | ACC={test_metrics['Accuracy']:.4f}"
        )
        print("Confusion matrix [[true0->pred0,1,2], ...]:", test_metrics["Confusion_Matrix"])

        row = {k: v for k, v in test_metrics.items() if k not in ("Confusion_Matrix", "Class_Report")}
        row["Fold"] = fold
        row["Best_Epoch"] = best_epoch
        fold_rows.append(row)

        pd.DataFrame(test_metrics["Class_Report"]).T.to_csv(
            os.path.join(fold_dir, "test_classification_report.csv")
        )
        with open(os.path.join(fold_dir, "test_metrics.txt"), "w") as f:
            f.write(str(test_metrics))

    result_df = pd.DataFrame(fold_rows)
    result_df.to_csv(os.path.join(args.save_dir, "10fold_results.csv"), index=False)

    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.drop("Fold")
    summary = pd.DataFrame({
        "mean": result_df[numeric_cols].mean(),
        "std": result_df[numeric_cols].std(),
    })
    summary.to_csv(os.path.join(args.save_dir, "10fold_summary.csv"))
    print("\n===== 10-fold summary =====")
    print(summary)
    print("Results saved to:", args.save_dir)


if __name__ == "__main__":
    main()
