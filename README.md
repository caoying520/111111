# -*- coding: utf-8 -*-

import pandas as pd
import openpyxl
import os
import re


# ============================================================
# 1. 路径设置
# ============================================================

# 原始预测结果 CSV
PRED_CSV = "/data/caoying/softwares/酶R2/prediction.csv"

# 原有 Excel 文件
INPUT_EXCEL = "/data/caoying/softwares/酶R2/original.xlsx"

# 修改后的 Excel 输出路径
OUTPUT_EXCEL = "/data/caoying/softwares/酶R2/original_R2.xlsx"

# 如果 Excel 的 sheet 名就是预测结果中的 SM，
# 保持 True
#
# 如果 sheet 名实际上对应 SMILE.csv 中的 Name，
# 则需要设置为 False，并使用下面的 SMILE_CSV。
SHEET_IS_SM = True

# 当 SHEET_IS_SM=False 时使用
SMILE_CSV = "/data/caoying/softwares/酶R2/SMILE.csv"


# ============================================================
# 2. 基础设置
# ============================================================

PRED_VALUE_COL = "pred_Conv"

PRED_RANK_COL = "rank"

EXCEL_VALUE_COL = "R2 Predicted Value"

EXCEL_RANK_COL = "R2 Predicted Rank"


# ============================================================
# 3. 清洗名称
# ============================================================

def clean_value(x):
    """
    用于匹配 Name / SM。
    防止字符串前后空格、数字格式等造成匹配失败。
    """
    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# 4. 读取预测 CSV
# ============================================================

print("=" * 60)
print("开始读取预测结果")
print("=" * 60)

if not os.path.exists(PRED_CSV):
    raise FileNotFoundError(
        f"找不到预测 CSV：\n{PRED_CSV}"
    )

pred_df = pd.read_csv(PRED_CSV)

print(f"预测结果数量：{len(pred_df)}")
print(f"预测 CSV 列：{list(pred_df.columns)}")


# ============================================================
# 5. 检查必要列
# ============================================================

required_cols = [
    "Name",
    "SM",
    PRED_VALUE_COL,
    PRED_RANK_COL
]

for col in required_cols:

    if col not in pred_df.columns:

        raise ValueError(
            f"预测 CSV 中缺少必要列：{col}\n"
            f"当前列为：{list(pred_df.columns)}"
        )


# ============================================================
# 6. 清洗 Name / SM
# ============================================================

pred_df["Name_match"] = pred_df["Name"].apply(clean_value)
pred_df["SM_match"] = pred_df["SM"].apply(clean_value)


# ============================================================
# 7. 检查重复 Name + SM
# ============================================================

duplicate_mask = pred_df.duplicated(
    subset=["Name_match", "SM_match"],
    keep=False
)

if duplicate_mask.any():

    duplicate_df = pred_df[
        duplicate_mask
    ][
        ["Name", "SM", PRED_VALUE_COL, PRED_RANK_COL]
    ]

    print("\n[WARNING] 发现重复的 Name + SM：")
    print(duplicate_df.to_string(index=False))

    raise ValueError(
        "\n同一个 Name + SM 出现多次，"
        "无法确定应该填哪一个预测结果。"
    )


# ============================================================
# 8. 建立预测结果字典
# ============================================================

# key:
#     (Name, SM)
#
# value:
#     (pred_Conv, rank)

prediction_dict = {}

for _, row in pred_df.iterrows():

    key = (
        row["Name_match"],
        row["SM_match"]
    )

    pred_value = row[PRED_VALUE_COL]
    pred_rank = row[PRED_RANK_COL]

    prediction_dict[key] = (
        pred_value,
        pred_rank
    )


print(
    f"\n建立预测结果映射：{len(prediction_dict)} 条"
)


# ============================================================
# 9. 如果 Sheet 名不是 SM，而是 Name
# ============================================================

sm_to_sheet_name = {}

if not SHEET_IS_SM:

    if not os.path.exists(SMILE_CSV):

        raise FileNotFoundError(
            f"找不到 SMILE CSV：\n{SMILE_CSV}"
        )

    smile_df = pd.read_csv(SMILE_CSV)

    if "SM" not in smile_df.columns:
        raise ValueError(
            "SMILE CSV 中缺少 SM 列"
        )

    if "Name" not in smile_df.columns:
        raise ValueError(
            "SMILE CSV 中缺少 Name 列"
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
# 10. 打开原 Excel
# ============================================================

if not os.path.exists(INPUT_EXCEL):

    raise FileNotFoundError(
        f"找不到原 Excel：\n{INPUT_EXCEL}"
    )

print("\n正在打开 Excel...")

wb = openpyxl.load_workbook(
    INPUT_EXCEL
)

print(
    f"Excel Sheet 数量：{len(wb.sheetnames)}"
)

print(
    "Sheet：",
    wb.sheetnames
)


# ============================================================
# 11. 处理每一个 Sheet
# ============================================================

total_updated = 0
total_not_found = 0
total_no_prediction = 0

sheet_statistics = []


for sheet_name in wb.sheetnames:

    ws = wb[sheet_name]

    print("\n" + "-" * 60)
    print(f"正在处理 Sheet：{sheet_name}")
    print("-" * 60)

    # --------------------------------------------------------
    # 确定当前 Sheet 对应的 SM
    # --------------------------------------------------------

    if SHEET_IS_SM:

        sheet_sm = clean_value(sheet_name)

    else:

        sheet_sm = None

        for sm, name in sm_to_sheet_name.items():

            if name == clean_value(sheet_name):

                sheet_sm = sm
                break

        if sheet_sm is None:

            print(
                f"[WARNING] Sheet {sheet_name} "
                f"无法在 SMILE.csv 中找到对应 SM"
            )

            continue

    # --------------------------------------------------------
    # 找到 Excel 中的表头
    # --------------------------------------------------------

    header_row = None

    name_col = None
    value_col = None
    rank_col = None

    # 默认扫描前 30 行寻找表头
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

            # 找 Name 列
            if cell_value == "Name":

                header_row = cell.row
                name_col = cell.column

            # 找 R2 Predicted Value
            if cell_value == EXCEL_VALUE_COL:

                header_row = cell.row
                value_col = cell.column

            # 找 R2 Predicted Rank
            if cell_value == EXCEL_RANK_COL:

                header_row = cell.row
                rank_col = cell.column

    # --------------------------------------------------------
    # 检查表头
    # --------------------------------------------------------

    if header_row is None:

        print(
            f"[WARNING] Sheet {sheet_name} "
            f"没有找到表头"
        )

        continue

    if name_col is None:

        print(
            f"[WARNING] Sheet {sheet_name} "
            f"没有找到 Name 列"
        )

        continue

    # --------------------------------------------------------
    # 如果没有 R2 Predicted Value，自动创建
    # --------------------------------------------------------

    if value_col is None:

        value_col = ws.max_column + 1

        ws.cell(
            row=header_row,
            column=value_col
        ).value = EXCEL_VALUE_COL

        print(
            f"自动创建列：{EXCEL_VALUE_COL}"
        )

    # --------------------------------------------------------
    # 如果没有 R2 Predicted Rank，自动创建
    # --------------------------------------------------------

    if rank_col is None:

        rank_col = ws.max_column + 1

        # 防止两列同时创建时位置问题
        if rank_col == value_col:

            rank_col += 1

        ws.cell(
            row=header_row,
            column=rank_col
        ).value = EXCEL_RANK_COL

        print(
            f"自动创建列：{EXCEL_RANK_COL}"
        )

    # --------------------------------------------------------
    # 当前 Sheet 的预测结果
    # --------------------------------------------------------

    sheet_updated = 0
    sheet_not_found = 0

    # --------------------------------------------------------
    # 遍历 Excel 每一行
    # --------------------------------------------------------

    for row_idx in range(
        header_row + 1,
        ws.max_row + 1
    ):

        name_cell = ws.cell(
            row=row_idx,
            column=name_col
        )

        name = clean_value(
            name_cell.value
        )

        if name == "":
            continue

        key = (
            name,
            sheet_sm
        )

        # ----------------------------------------------------
        # 查找预测结果
        # ----------------------------------------------------

        if key not in prediction_dict:

            sheet_not_found += 1

            continue

        pred_value, pred_rank = (
            prediction_dict[key]
        )

        # ----------------------------------------------------
        # 写入 Predicted Value
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
        # 写入 Predicted Rank
        # ----------------------------------------------------

        rank_cell = ws.cell(
            row=row_idx,
            column=rank_col
        )

        if pd.isna(pred_rank):

            rank_cell.value = None

        else:

            # rank 通常应该是整数
            try:

                rank_cell.value = int(
                    float(pred_rank)
                )

            except:

                rank_cell.value = pred_rank

        sheet_updated += 1
        total_updated += 1

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    total_not_found += sheet_not_found

    sheet_statistics.append(
        {
            "Sheet": sheet_name,
            "SM": sheet_sm,
            "Updated": sheet_updated,
            "Not_Found": sheet_not_found
        }
    )

    print(
        f"Sheet 更新：{sheet_updated}"
    )

    print(
        f"Sheet 未找到预测结果：{sheet_not_found}"
    )


# ============================================================
# 12. 保存 Excel
# ============================================================

print("\n" + "=" * 60)
print("正在保存 Excel")
print("=" * 60)

# 如果输出目录不存在，创建目录
output_dir = os.path.dirname(
    OUTPUT_EXCEL
)

if output_dir != "":
    os.makedirs(
        output_dir,
        exist_ok=True
    )

wb.save(
    OUTPUT_EXCEL
)


# ============================================================
# 13. 输出统计
# ============================================================

print("\n" + "=" * 60)
print("处理完成")
print("=" * 60)

print(
    f"原始 Excel：{INPUT_EXCEL}"
)

print(
    f"输出 Excel：{OUTPUT_EXCEL}"
)

print(
    f"总共写入预测结果：{total_updated}"
)

print(
    f"没有找到对应预测结果：{total_not_found}"
)

print(
    f"Sheet 数量：{len(wb.sheetnames)}"
)

print("\n各 Sheet 统计：")

for item in sheet_statistics:

    print(
        f"{item['Sheet']:<30} "
        f"更新={item['Updated']:<6} "
        f"未找到={item['Not_Found']}"
    )

print("\n完成！")
