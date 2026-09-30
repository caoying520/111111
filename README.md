# -*- coding: utf-8 -*-

import pandas as pd
import openpyxl
import os


# ============================================================
# 1. 路径设置
# ============================================================

# 预测结果 CSV
PRED_CSV = "/data/caoying/softwares/酶R2/prediction.csv"

# 原始 Excel 模板
INPUT_EXCEL = "/data/caoying/softwares/酶R2/original.xlsx"

# 输出 Excel
OUTPUT_EXCEL = "/data/caoying/softwares/酶R2/original_R2.xlsx"

# ============================================================
# Excel Sheet 与 SM 的关系
#
# True：
#     Excel Sheet 名 = SM
#
# False：
#     Excel Sheet 名 = SMILE.csv 中的 Name
# ============================================================

SHEET_IS_SM = True

# 如果 SHEET_IS_SM = False 才需要
SMILE_CSV = "/data/caoying/softwares/酶R2/SMILE.csv"


# ============================================================
# 2. 参数
# ============================================================

# False = pred_Conv 从大到小
# True  = pred_Conv 从小到大
SORT_ASCENDING = False

# ============================================================
# Excel 模板中的实际列名
# ============================================================

EXCEL_ID_COL = "Biortus No."

EXCEL_VALUE_COL = "R2 Predicted Value"

EXCEL_RANK_COL = "R2 Predicted Rank"


# ============================================================
# 3. 字符串清洗
# ============================================================

def clean_value(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# 4. 读取预测结果
# ============================================================

print("=" * 70)
print("读取预测结果")
print("=" * 70)

if not os.path.exists(PRED_CSV):
    raise FileNotFoundError(
        f"找不到预测 CSV：\n{PRED_CSV}"
    )

df = pd.read_csv(PRED_CSV)

print(f"预测数据数量：{len(df)}")
print("CSV 列：")
print(list(df.columns))


# ============================================================
# 5. 检查预测 CSV 必要列
# ============================================================

required_cols = [
    "Name",
    "SM",
    "pred_Conv"
]

for col in required_cols:

    if col not in df.columns:

        raise ValueError(
            f"\n预测 CSV 缺少必要列：{col}\n"
            f"当前 CSV 列：{list(df.columns)}"
        )


# ============================================================
# 6. 清洗 Name / SM
# ============================================================

df["Name_match"] = (
    df["Name"].apply(clean_value)
)

df["SM_match"] = (
    df["SM"].apply(clean_value)
)


# ============================================================
# 7. 按 SM 分组计算 rank
# ============================================================

print("\n" + "=" * 70)
print("按照 SM 分组计算 rank")
print("=" * 70)

ranked_groups = []

for sm_value, group in df.groupby(
    "SM_match",
    sort=False
):

    # --------------------------------------------------------
    # 按 pred_Conv 排序
    # --------------------------------------------------------

    sorted_group = group.sort_values(
        by="pred_Conv",
        ascending=SORT_ASCENDING
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 生成 rank
    # --------------------------------------------------------

    sorted_group["rank"] = (
        sorted_group.index + 1
    )

    ranked_groups.append(
        sorted_group
    )


# 合并所有 SM
ranked_df = pd.concat(
    ranked_groups,
    ignore_index=True
)

print(
    f"SM 数量：{ranked_df['SM_match'].nunique()}"
)

print(
    f"总预测数量：{len(ranked_df)}"
)


# ============================================================
# 8. 显示 rank 示例
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
# 9. 建立 Name + SM → prediction 映射
# ============================================================

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
    f"\n预测映射数量：{len(prediction_dict)}"
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
# 11. 打开 Excel
# ============================================================

if not os.path.exists(INPUT_EXCEL):

    raise FileNotFoundError(
        f"找不到 Excel：\n{INPUT_EXCEL}"
    )

print("\n" + "=" * 70)
print("打开 Excel 模板")
print("=" * 70)

wb = openpyxl.load_workbook(
    INPUT_EXCEL
)

print(
    f"Sheet 数量：{len(wb.sheetnames)}"
)

print(
    "Sheet 名称："
)

for sheet_name in wb.sheetnames:

    print(
        f"  {sheet_name}"
    )


# ============================================================
# 12. 统计
# ============================================================

total_updated = 0

total_not_found = 0

total_sheet_not_found = 0


# ============================================================
# 13. 遍历 Sheet
# ============================================================

for sheet_name in wb.sheetnames:

    ws = wb[sheet_name]

    print("\n" + "-" * 70)
    print(f"处理 Sheet：{sheet_name}")
    print("-" * 70)


    # ========================================================
    # 确定当前 Sheet 对应的 SM
    # ========================================================

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
                "[WARNING] "
                "无法找到该 Sheet 对应的 SM"
            )

            total_sheet_not_found += 1

            continue


    # ========================================================
    # 查找表头
    # ========================================================

    header_row = None

    biortus_col = None

    value_col = None

    rank_col = None


    # 在前 30 行寻找表头
    for row in ws.iter_rows(
        min_row=1,
        max_row=min(ws.max_row, 30)
    ):

        for cell in row:

            if cell.value is None:
                continue

            cell_value = str(
                cell.value
            ).strip()


            # ------------------------------------------------
            # 找 Biortus No.
            # ------------------------------------------------

            if cell_value == EXCEL_ID_COL:

                header_row = cell.row

                biortus_col = cell.column


            # ------------------------------------------------
            # 找 R2 Predicted Value
            # ------------------------------------------------

            if cell_value == EXCEL_VALUE_COL:

                header_row = cell.row

                value_col = cell.column


            # ------------------------------------------------
            # 找 R2 Predicted Rank
            # ------------------------------------------------

            if cell_value == EXCEL_RANK_COL:

                header_row = cell.row

                rank_col = cell.column


    # ========================================================
    # 检查 Biortus No.
    # ========================================================

    if biortus_col is None:

        print(
            f"[WARNING] Sheet '{sheet_name}' "
            f"没有找到 '{EXCEL_ID_COL}' 列"
        )

        continue


    print(
        f"找到 {EXCEL_ID_COL}："
        f"第 {biortus_col} 列"
    )


    # ========================================================
    # 如果没有 R2 Predicted Value
    # 自动创建
    # ========================================================

    if value_col is None:

        value_col = ws.max_column + 1

        ws.cell(
            row=header_row,
            column=value_col
        ).value = EXCEL_VALUE_COL

        print(
            f"创建列：{EXCEL_VALUE_COL}"
        )


    # ========================================================
    # 如果没有 R2 Predicted Rank
    # 自动创建
    # ========================================================

    if rank_col is None:

        rank_col = ws.max_column + 1

        # 防止与 value_col 重叠
        if rank_col == value_col:

            rank_col += 1

        ws.cell(
            row=header_row,
            column=rank_col
        ).value = EXCEL_RANK_COL

        print(
            f"创建列：{EXCEL_RANK_COL}"
        )


    # ========================================================
    # 开始填充
    # ========================================================

    sheet_updated = 0

    sheet_not_found = 0


    for row_idx in range(
        header_row + 1,
        ws.max_row + 1
    ):


        # ----------------------------------------------------
        # 读取 Biortus No.
        # ----------------------------------------------------

        biortus_no = ws.cell(
            row=row_idx,
            column=biortus_col
        ).value

        biortus_no = clean_value(
            biortus_no
        )


        if biortus_no == "":
            continue


        # ====================================================
        # 关键匹配：
        #
        # Excel：
        #     Biortus No.
        #
        # CSV：
        #     Name
        #
        # 同时：
        #     Excel Sheet → SM
        #
        # ====================================================

        key = (
            biortus_no,
            sheet_sm
        )


        # ----------------------------------------------------
        # 查找预测结果
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


        # ====================================================
        # 写入 R2 Predicted Value
        # ====================================================

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


        # ====================================================
        # 写入 R2 Predicted Rank
        # ====================================================

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


    # ========================================================
    # 当前 Sheet 统计
    # ========================================================

    print(
        f"成功填入：{sheet_updated}"
    )

    print(
        f"未找到匹配：{sheet_not_found}"
    )


# ============================================================
# 14. 保存
# ============================================================

print("\n" + "=" * 70)
print("保存 Excel")
print("=" * 70)


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

print("\n" + "=" * 70)
print("处理完成")
print("=" * 70)

print(
    f"输入 Excel：\n{INPUT_EXCEL}"
)

print(
    f"\n输出 Excel：\n{OUTPUT_EXCEL}"
)

print(
    f"\n成功填入：{total_updated}"
)

print(
    f"未匹配：{total_not_found}"
)

print(
    f"找不到对应 Sheet：{total_sheet_not_found}"
)

print("=" * 70)
