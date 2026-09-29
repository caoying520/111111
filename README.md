# -*- coding: utf-8 -*-

import pandas as pd
import openpyxl
import os


# ============================================================
# 1. 路径
# ============================================================

# 预测结果 CSV
PRED_CSV = "/data/caoying/softwares/酶R2/prediction.csv"

# 原始 Excel
INPUT_EXCEL = "/data/caoying/softwares/酶R2/original.xlsx"

# 输出 Excel
OUTPUT_EXCEL = "/data/caoying/softwares/酶R2/original_R2.xlsx"

# ------------------------------------------------------------
# Excel 的 Sheet 名是否就是 SM？
#
# True：
#     Sheet 名 = SM
#
# False：
#     Sheet 名 = SMILE.csv 中的 Name
#     此时需要填写 SMILE_CSV
# ------------------------------------------------------------

SHEET_IS_SM = True

SMILE_CSV = "/data/caoying/softwares/酶R2/SMILE.csv"


# ============================================================
# 2. 参数
# ============================================================

# pred_Conv 越大排名越靠前
# 与你之前代码默认行为一致
SORT_ASCENDING = False

# Excel 中要写入的列
EXCEL_VALUE_COL = "R2 Predicted Value"
EXCEL_RANK_COL = "R2 Predicted Rank"


# ============================================================
# 3. 清洗字符串
# ============================================================

def clean_value(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# 4. 读取预测 CSV
# ============================================================

print("=" * 60)
print("读取预测结果")
print("=" * 60)

if not os.path.exists(PRED_CSV):
    raise FileNotFoundError(
        f"找不到预测 CSV：\n{PRED_CSV}"
    )

df = pd.read_csv(PRED_CSV)

print(f"预测结果数量：{len(df)}")
print(f"CSV 列：{list(df.columns)}")


# ============================================================
# 5. 检查必要列
# ============================================================

required_cols = [
    "Name",
    "SM",
    "pred_Conv"
]

for col in required_cols:

    if col not in df.columns:

        raise ValueError(
            f"预测 CSV 缺少必要列：{col}\n"
            f"当前列：{list(df.columns)}"
        )


# ============================================================
# 6. 清洗 Name / SM
# ============================================================

df["Name_match"] = df["Name"].apply(clean_value)
df["SM_match"] = df["SM"].apply(clean_value)


# ============================================================
# 7. 按 SM 分组，并重新计算 rank
# ============================================================

print("\n正在按照 SM 计算 rank...")

# ------------------------------------------------------------
# 这里就是你之前代码中的核心逻辑
#
# 每一个 SM 单独排名：
#
# pred_Conv 最大 → rank 1
# 第二大       → rank 2
# 第三大       → rank 3
# ...
#
# 如果 SORT_ASCENDING=True：
#
# pred_Conv 最小 → rank 1
# ------------------------------------------------------------

ranked_groups = []

for sm_value, group in df.groupby(
    "SM_match",
    sort=False
):

    # 按 pred_Conv 排序
    sorted_group = group.sort_values(
        by="pred_Conv",
        ascending=SORT_ASCENDING
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 重新生成 rank
    # --------------------------------------------------------

    sorted_group["rank"] = (
        sorted_group.index + 1
    )

    ranked_groups.append(
        sorted_group
    )


# 合并
ranked_df = pd.concat(
    ranked_groups,
    ignore_index=True
)

print(
    f"共 {ranked_df['SM_match'].nunique()} 个不同 SM"
)

print(
    f"共生成 {len(ranked_df)} 条排名结果"
)


# ============================================================
# 8. 检查 rank 是否生成成功
# ============================================================

print("\nRank 示例：")

print(
    ranked_df[
        [
            "Name",
            "SM",
            "pred_Conv",
            "rank"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 9. 建立预测结果字典
# ============================================================

# 使用：
#
# Name + SM
#
# 作为唯一匹配条件

prediction_dict = {}

for _, row in ranked_df.iterrows():

    key = (
        clean_value(row["Name"]),
        clean_value(row["SM"])
    )

    prediction_dict[key] = {
        "pred_Conv": row["pred_Conv"],
        "rank": row["rank"]
    }


print(
    f"\n建立预测映射：{len(prediction_dict)} 条"
)


# ============================================================
# 10. 如果 Sheet 名不是 SM
# ============================================================

sm_to_sheet_name = {}

if not SHEET_IS_SM:

    if not os.path.exists(SMILE_CSV):

        raise FileNotFoundError(
            f"找不到 SMILE.csv：\n{SMILE_CSV}"
        )

    smile_df = pd.read_csv(
        SMILE_CSV
    )

    if "SM" not in smile_df.columns:

        raise ValueError(
            "SMILE.csv 中没有 SM 列"
        )

    if "Name" not in smile_df.columns:

        raise ValueError(
            "SMILE.csv 中没有 Name 列"
        )

    smile_df["SM_match"] = (
        smile_df["SM"].apply(clean_value)
    )

    smile_df["Name_match"] = (
        smile_df["Name"].apply(clean_value)
    )

    sm_to_sheet_name = dict(
        zip(
            smile_df["SM_match"],
            smile_df["Name_match"]
        )
    )


# ============================================================
# 11. 打开原 Excel
# ============================================================

if not os.path.exists(INPUT_EXCEL):

    raise FileNotFoundError(
        f"找不到 Excel：\n{INPUT_EXCEL}"
    )

print("\n正在打开 Excel...")

wb = openpyxl.load_workbook(
    INPUT_EXCEL
)

print(
    f"Excel 中共有 {len(wb.sheetnames)} 个 Sheet"
)


# ============================================================
# 12. 统计
# ============================================================

total_updated = 0
total_not_found = 0


# ============================================================
# 13. 遍历 Excel Sheet
# ============================================================

for sheet_name in wb.sheetnames:

    ws = wb[sheet_name]

    print("\n" + "-" * 60)
    print(f"处理 Sheet：{sheet_name}")
    print("-" * 60)

    # --------------------------------------------------------
    # 确定这个 Sheet 对应哪个 SM
    # --------------------------------------------------------

    if SHEET_IS_SM:

        sheet_sm = clean_value(
            sheet_name
        )

    else:

        sheet_sm = None

        sheet_name_clean = clean_value(
            sheet_name
        )

        for sm, name in sm_to_sheet_name.items():

            if name == sheet_name_clean:

                sheet_sm = sm
                break

        if sheet_sm is None:

            print(
                "[WARNING] 无法找到该 Sheet 对应的 SM"
            )

            continue

    # --------------------------------------------------------
    # 查找表头
    # --------------------------------------------------------

    header_row = None
    name_col = None
    value_col = None
    rank_col = None

    # 搜索前 30 行
    for row in ws.iter_rows(
        min_row=1,
        max_row=min(ws.max_row, 30)
    ):

        for cell in row:

            if cell.value is None:
                continue

            value = str(
                cell.value
            ).strip()

            if value == "Name":

                header_row = cell.row
                name_col = cell.column

            elif value == EXCEL_VALUE_COL:

                header_row = cell.row
                value_col = cell.column

            elif value == EXCEL_RANK_COL:

                header_row = cell.row
                rank_col = cell.column


    # --------------------------------------------------------
    # 没有 Name
    # --------------------------------------------------------

    if name_col is None:

        print(
            "[WARNING] 没有找到 Name 列"
        )

        continue


    # --------------------------------------------------------
    # 如果没有 R2 Predicted Value
    # 自动创建
    # --------------------------------------------------------

    if value_col is None:

        value_col = ws.max_column + 1

        ws.cell(
            row=header_row,
            column=value_col
        ).value = EXCEL_VALUE_COL

        print(
            f"创建列：{EXCEL_VALUE_COL}"
        )


    # --------------------------------------------------------
    # 如果没有 R2 Predicted Rank
    # 自动创建
    # --------------------------------------------------------

    if rank_col is None:

        rank_col = ws.max_column + 1

        # 防止两个新列发生位置冲突
        if rank_col == value_col:

            rank_col += 1

        ws.cell(
            row=header_row,
            column=rank_col
        ).value = EXCEL_RANK_COL

        print(
            f"创建列：{EXCEL_RANK_COL}"
        )


    # --------------------------------------------------------
    # 开始填充
    # --------------------------------------------------------

    sheet_updated = 0
    sheet_not_found = 0

    for row_idx in range(
        header_row + 1,
        ws.max_row + 1
    ):

        name = clean_value(
            ws.cell(
                row=row_idx,
                column=name_col
            ).value
        )

        if name == "":
            continue

        # ----------------------------------------------------
        # Name + SM
        # ----------------------------------------------------

        key = (
            name,
            sheet_sm
        )

        # ----------------------------------------------------
        # 查找预测
        # ----------------------------------------------------

        if key not in prediction_dict:

            sheet_not_found += 1
            total_not_found += 1

            continue

        prediction = (
            prediction_dict[key]
        )

        pred_value = prediction[
            "pred_Conv"
        ]

        pred_rank = prediction[
            "rank"
        ]


        # ----------------------------------------------------
        # 写 R2 Predicted Value
        # ----------------------------------------------------

        value_cell = ws.cell(
            row=row_idx,
            column=value_col
        )

        if pd.isna(pred_value):

            value_cell.value = None

        else:

            value_cell.value = round(
                float(pred_value),
                3
            )


        # ----------------------------------------------------
        # 写 R2 Predicted Rank
        # ----------------------------------------------------

        rank_cell = ws.cell(
            row=row_idx,
            column=rank_col
        )

        if pd.isna(pred_rank):

            rank_cell.value = None

        else:

            rank_cell.value = int(
                float(pred_rank)
            )


        sheet_updated += 1
        total_updated += 1


    print(
        f"成功填入：{sheet_updated}"
    )

    print(
        f"没有匹配到：{sheet_not_found}"
    )


# ============================================================
# 14. 保存
# ============================================================

print("\n" + "=" * 60)
print("保存 Excel")
print("=" * 60)

output_dir = os.path.dirname(
    OUTPUT_EXCEL
)

if output_dir:

    os.makedirs(
        output_dir,
        exist_ok=True
    )

wb.save(
    OUTPUT_EXCEL
)


# ============================================================
# 15. 最终统计
# ============================================================

print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)

print(
    f"输入 Excel：{INPUT_EXCEL}"
)

print(
    f"输出 Excel：{OUTPUT_EXCEL}"
)

print(
    f"总填入数量：{total_updated}"
)

print(
    f"未匹配数量：{total_not_found}"
)

print("=" * 60)
