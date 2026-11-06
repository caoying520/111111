
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# =========================
# 1. 路径和参数
# =========================
INPUT_CSV = "train_data.csv"       # 原始完整数据
OUTPUT_DIR = "data_split"          # 划分结果保存目录

GROUP_COL = "Name"                 # 酶的唯一标识
LABEL_COL = "Activity"             # 原始活性标签

ACTIVITY_THRESHOLD = 0.03
RANDOM_STATE = 123
N_SPLITS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV)

required_cols = [GROUP_COL, LABEL_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"数据中缺少必要列: {col}")

# 删除酶标识或活性值缺失的记录
df = df.dropna(subset=required_cols).copy()
df = df.reset_index(drop=True)

# 确保 Activity 为数值
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

# =========================
# 3. 构造二分类标签
# =========================
# Activity >= 0.03：有活性
# Activity < 0.03：无活性
y = (df[LABEL_COL].values >= ACTIVITY_THRESHOLD).astype(int)

# 按酶分组
groups = df[GROUP_COL].astype(str).values

print("=" * 60)
print("原始数据统计")
print("=" * 60)
print(f"总样本数: {len(df)}")
print(f"酶的数量: {df[GROUP_COL].nunique()}")
print(f"无活性样本: {(y == 0).sum()}")
print(f"有活性样本: {(y == 1).sum()}")
print(f"有活性比例: {(y == 1).mean():.4f}")

# =========================
# 4. 分层分组划分
# =========================
# 20折，每折约5%
# Fold 0-13：训练集，约70%
# Fold 14-16：验证集，约15%
# Fold 17-19：测试集，约15%

sgkf = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_ids = np.full(len(df), -1, dtype=int)

for fold, (_, val_idx) in enumerate(
    sgkf.split(df, y, groups)
):
    fold_ids[val_idx] = fold

if np.any(fold_ids == -1):
    raise RuntimeError("部分样本没有被分配到数据集！")

train_idx = np.where(fold_ids < 14)[0]
val_idx = np.where((fold_ids >= 14) & (fold_ids < 17))[0]
test_idx = np.where(fold_ids >= 17)[0]

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

# =========================
# 5. 检查酶是否存在交叉
# =========================
train_groups = set(train_df[GROUP_COL].astype(str))
val_groups = set(val_df[GROUP_COL].astype(str))
test_groups = set(test_df[GROUP_COL].astype(str))

assert len(train_groups & val_groups) == 0, "训练集和验证集存在相同酶！"
assert len(train_groups & test_groups) == 0, "训练集和测试集存在相同酶！"
assert len(val_groups & test_groups) == 0, "验证集和测试集存在相同酶！"

print("\n酶分组交叉检查通过：三个数据集之间没有相同的酶。")

# =========================
# 6. 保存三个数据集
# =========================
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path = os.path.join(OUTPUT_DIR, "val.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# =========================
# 7. 输出数据集统计
# =========================
def get_summary(name, data):
    labels = (data[LABEL_COL].values >= ACTIVITY_THRESHOLD).astype(int)

    return {
        "Dataset": name,
        "Samples": len(data),
        "Enzymes": data[GROUP_COL].nunique(),
        "Inactive": int((labels == 0).sum()),
        "Active": int((labels == 1).sum()),
        "Active_Ratio": float((labels == 1).mean()),
        "Activity_Mean": float(data[LABEL_COL].mean()),
        "Activity_Median": float(data[LABEL_COL].median())
    }

summary = pd.DataFrame([
    get_summary("Train", train_df),
    get_summary("Validation", val_df),
    get_summary("Test", test_df)
])

summary.to_csv(
    os.path.join(OUTPUT_DIR, "split_summary.csv"),
    index=False
)

print("\n" + "=" * 80)
print("数据划分结果")
print("=" * 80)
print(summary.to_string(index=False))

# =========================
# 8. 保存酶分组信息
# =========================
split_groups = {
    "train": sorted(train_groups),
    "val": sorted(val_groups),
    "test": sorted(test_groups)
}

with open(
    os.path.join(OUTPUT_DIR, "split_groups.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(split_groups, f, ensure_ascii=False, indent=4)

print("\n文件保存完成：")
print(f"训练集: {train_path}")
print(f"验证集: {val_path}")
print(f"测试集: {test_path}")
print(f"统计表: {OUTPUT_DIR}/split_summary.csv")
print(f"分组记录: {OUTPUT_DIR}/split_groups.json")
