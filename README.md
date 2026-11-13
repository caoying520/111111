
# -*- coding: utf-8 -*-

import os
from copy import copy
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


# ============================================================
# 文件路径
# ============================================================
input_xlsx = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/110KRD_96SM_std.xlsx"

output_xlsx = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/110KRD_96SM_std_format.xlsx"


# ============================================================
# 颜色设置
# ============================================================

# 排名1：深绿色
dark_green = (0, 97, 0)

# 排名30：浅绿色
light_green = (226, 240, 217)


# ============================================================
# 绿色渐变计算
# ============================================================
def get_green_color(rank):

    ratio = (rank - 1) / 29

    r = int(dark_green[0] + (light_green[0] - dark_green[0]) * ratio)
    g = int(dark_green[1] + (light_green[1] - dark_green[1]) * ratio)
    b = int(dark_green[2] + (light_green[2] - dark_green[2]) * ratio)

    return "{:02X}{:02X}{:02X}".format(r, g, b)


# ============================================================
# 读取Excel
# ============================================================
wb = load_workbook(input_xlsx)

total_sheets = 0
success_sheets = 0
total_colored = 0


# ============================================================
# 遍历所有Sheet
# ============================================================
for ws in wb.worksheets:

    total_sheets += 1

    print("\n===================================")
    print("Sheet:", ws.title)

    # --------------------------------------------------------
    # 1. 查找表头
    # --------------------------------------------------------
    header_row = None

    for r in range(1, min(ws.max_row, 30) + 1):

        values = [
            str(ws.cell(r, c).value).strip()
            if ws.cell(r, c).value is not None else ""
            for c in range(1, ws.max_column + 1)
        ]

        if "Biortus No." in values:
            header_row = r
            break

    if header_row is None:
        print("[跳过] 没有找到表头")
        continue


    # --------------------------------------------------------
    # 2. 找到所有R2预测列
    # --------------------------------------------------------
    pred_cols = []
    rank_cols = []

    for c in range(1, ws.max_column + 1):

        value = ws.cell(header_row, c).value

        if value is None:
            continue

        value = str(value).strip()

        if value == "R2 Predicted Value":
            pred_cols.append(c)

        elif value == "R2 Predicted Rank":
            rank_cols.append(c)


    if not pred_cols or not rank_cols:
        print("[跳过] 没有找到R2预测列")
        continue


    # 默认取最后一组新增列
    pred_col = pred_cols[-1]
    rank_col = rank_cols[-1]

    print("预测值列:", pred_col)
    print("排名列:", rank_col)


    # --------------------------------------------------------
    # 3. 设置表头格式
    #    复制其他正常表头的格式
    # --------------------------------------------------------

    # 以Biortus No.表头作为格式模板
    template_col = None

    for c in range(1, ws.max_column + 1):

        if ws.cell(header_row, c).value == "Biortus No.":
            template_col = c
            break

    if template_col is not None:

        template_cell = ws.cell(header_row, template_col)

        for target_col in [pred_col, rank_col]:

            target_cell = ws.cell(header_row, target_col)

            target_cell.font = copy(template_cell.font)
            target_cell.fill = copy(template_cell.fill)
            target_cell.border = copy(template_cell.border)
            target_cell.alignment = copy(template_cell.alignment)
            target_cell.protection = copy(template_cell.protection)
            target_cell.number_format = template_cell.number_format

    # --------------------------------------------------------
    # 4. 排名列设置绿色渐变
    #    Rank 1最深，Rank 30最浅
    # --------------------------------------------------------

    colored_count = 0

    for r in range(header_row + 1, ws.max_row + 1):

        cell = ws.cell(r, rank_col)

        if cell.value is None:
            continue

        try:
            rank = int(float(cell.value))
        except (ValueError, TypeError):
            continue

        # 只处理排名前30，包括30
        if 1 <= rank <= 30:

            color = get_green_color(rank)

            cell.fill = PatternFill(
                fill_type="solid",
                fgColor=color
            )

            colored_count += 1

    total_colored += colored_count
    success_sheets += 1

    print("设置绿色排名数量:", colored_count)


# ============================================================
# 5. 保存
# ============================================================
wb.save(output_xlsx)

print("\n===================================")
print("Excel格式处理完成")
print("===================================")

print("Sheet总数:", total_sheets)
print("成功处理Sheet:", success_sheets)
print("设置绿色排名总数:", total_colored)
print("输出文件:", output_xlsx)
