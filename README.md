
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 123
N_TRIALS = 2000

CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/train_data3.csv"
OUTPUT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/split_Activity_enz2"

ID_COL = "idx"
GROUP_COL = "Name"       # 按酶分组
# GROUP_COL = "SM"       # 如果要按底物分组，改成这一列

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(CSV_PATH)
df[ID_COL] = df[ID_COL].astype(str).str.strip()
df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")

df = df.dropna(subset=[ID_COL, GROUP_COL, "Activity"]).reset_index(drop=True)

if df[ID_COL].duplicated().any():
    raise ValueError("idx 存在重复，请先检查数据。")

# Activity 分箱：保留低活性和高活性样本的分布信息
bins = [-np.inf, 0, 0.03, 1, 5, 10, np.inf]
labels = ["<=0", "0-0.03", "0.03-1", "1-5", "5-10", ">=10"]
df["Activity_bin"] = pd.cut(
    df["Activity"], bins=bins, labels=labels,
    right=False, include_lowest=True
)

# 整体 Activity 分布
global_dist = df["Activity_bin"].value_counts(normalize=True).reindex(labels, fill_value=0)

def split_score(parts):
    """越小越好：同时考虑活性分布、数据量和酶数量"""
    target_rows = [0.8, 0.1, 0.1]
    total_rows = len(df)
    total_groups = df[GROUP_COL].nunique()

    score = 0.0

    for i, part in enumerate(parts):
        # 各 Activity 区间占比与整体占比的差异
        dist = part["Activity_bin"].value_counts(normalize=True).reindex(labels, fill_value=0)
        dist_error = np.abs(dist.values - global_dist.values).mean()

        # 样本数量比例误差
        row_error = abs(len(part) / total_rows - target_rows[i])

        # 酶数量比例误差
        group_error = abs(part[GROUP_COL].nunique() / total_groups - target_rows[i])

        score += dist_error + 0.5 * row_error + 0.2 * group_error

    return score

# 多次随机搜索，选择最平衡的酶分组方案
best_score = np.inf
best_parts = None

for trial in range(N_TRIALS):
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED + trial
    )
    train_pos, temp_pos = next(gss1.split(df, groups=df[GROUP_COL]))

    df_train = df.iloc[train_pos].copy()
    df_temp = df.iloc[temp_pos].copy()

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED + trial + 100000
    )
    val_pos, test_pos = next(
        gss2.split(df_temp, groups=df_temp[GROUP_COL])
    )

    df_val = df_temp.iloc[val_pos].copy()
    df_test = df_temp.iloc[test_pos].copy()

    parts = [df_train, df_val, df_test]
    score = split_score(parts)

    if score < best_score:
        best_score = score
        best_parts = [x.copy() for x in parts]

df_train, df_val, df_test = best_parts

# 检查酶是否跨集合
g_train = set(df_train[GROUP_COL])
g_val = set(df_val[GROUP_COL])
g_test = set(df_test[GROUP_COL])

assert not (g_train & g_val)
assert not (g_train & g_test)
assert not (g_val & g_test)

# 保存 idx TXT
for name, part in [
    ("train.txt", df_train),
    ("val.txt", df_val),
    ("test.txt", df_test)
]:
    part[ID_COL].to_csv(
        os.path.join(OUTPUT_DIR, name),
        index=False,
        header=False
    )

# 输出统计信息
summary = []
for name, part in [
    ("train", df_train),
    ("val", df_val),
    ("test", df_test)
]:
    row = {
        "Split": name,
        "Rows": len(part),
        "Enzymes": part[GROUP_COL].nunique(),
        "Activity_mean": part["Activity"].mean(),
        "Activity_std": part["Activity"].std(),
        "Activity_min": part["Activity"].min(),
        "Activity_max": part["Activity"].max()
    }

    for label in labels:
        row[f"{label}_count"] = (part["Activity_bin"] == label).sum()
        row[f"{label}_ratio"] = (part["Activity_bin"] == label).mean()

    summary.append(row)

summary = pd.DataFrame(summary)
summary.to_csv(
    os.path.join(OUTPUT_DIR, "split_summary.csv"),
    index=False
)

print("最佳划分分数:", best_score)
print(summary.to_string(index=False))
print("酶分组 + Activity 分布平衡划分完成:", OUTPUT_DIR)
