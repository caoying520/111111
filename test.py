
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 123
CSV_PATH = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/train_data3.csv"
OUTPUT_DIR = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/split_Activity_substrate"

ID_COL = "idx"
GROUP_COL = "SM"

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(CSV_PATH)
df[ID_COL] = df[ID_COL].astype(str)
df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
df = df.dropna(subset=[ID_COL, GROUP_COL]).reset_index(drop=True)

# 要求每行 idx 唯一，避免 TXT 无法准确映射
if df[ID_COL].duplicated().any():
    raise ValueError("idx 存在重复，请先确认 idx 是否为每条实验记录的唯一编号。")

# 第一步：80% train，20% 临时集合
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_pos, temp_pos = next(gss1.split(df, groups=df[GROUP_COL]))

df_train = df.iloc[train_pos].copy()
df_temp = df.iloc[temp_pos].copy()

# 第二步：临时集合一半作为 val，一半作为 test
# 相当于全数据约 80% / 10% / 10%
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_pos, test_pos = next(gss2.split(df_temp, groups=df_temp[GROUP_COL]))

df_val = df_temp.iloc[val_pos].copy()
df_test = df_temp.iloc[test_pos].copy()

# 检查底物是否跨集合
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

# 输出统计
summary = []
for name, part in [
    ("train", df_train),
    ("val", df_val),
    ("test", df_test)
]:
    summary.append({
        "Split": name,
        "Rows": len(part),
        "Substrates": part[GROUP_COL].nunique(),
        "Enzymes": part["Name"].nunique(),
        "Activity_0": (part["Activity"] == 0).sum(),
        "Activity_mean": part["Activity"].mean(),
        "Activity_std": part["Activity"].std()
    })

summary = pd.DataFrame(summary)
summary.to_csv(
    os.path.join(OUTPUT_DIR, "split_summary.csv"),
    index=False
)

print(summary.to_string(index=False))
print("底物分组划分完成:", OUTPUT_DIR)
