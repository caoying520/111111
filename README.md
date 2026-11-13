
# -*- coding: utf-8 -*-

import pandas as pd
from openpyxl import load_workbook

pred_csv = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/predictions_110KRD_SM96_std/best_model_add_preds.csv"
smile_csv = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/96SM_std.csv"
template_xlsx = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/110KRD&96底物实验结果与多模型综合分析_结果.xlsx"
output_xlsx = "/data/caoying/softwares/酶R2/enzyme_交接20260609/new_data_test/110KRD_96SM_std.xlsx"


# ============================================================
# 清洗编号
# ============================================================
def clean_id(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def clean_sm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# ============================================================
# 1. 读取预测结果
# ============================================================
pred_df = pd.read_csv(pred_csv)

pred_df["SM_match"] = pred_df["SM"].apply(clean_sm)
pred_df["Name_match"] = pred_df["Name"].apply(clean_id)


# ============================================================
# 2. SM → Sheet名称
# ============================================================
smile_df = pd.read_csv(smile_csv)

smile_df["SM_match"] = smile_df["SM"].apply(clean_sm)
smile_df["Sheet_Name"] = smile_df["Name"].astype(str).str.strip()

sm_to_sheet = dict(zip(smile_df["SM_match"], smile_df["Sheet_Name"]))

pred_df["Sheet_Name"] = pred_df["SM_match"].map(sm_to_sheet)


# ============================================================
# 3. 每个底物内部计算排名
# ============================================================
pred_df["rank"] = (
    pred_df.groupby("SM_match")["pred_Conv"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)


# ============================================================
# 4. 打开 Excel 模板
# ============================================================
wb = load_workbook(template_xlsx)

total_match = 0
total_unmatch = 0
sheet_match = 0
sheet_not_found = 0


# ============================================================
# 5. 按底物 Sheet 处理
# ============================================================
for sheet_name, group in pred_df.groupby("Sheet_Name", dropna=True):

    sheet_name = str(sheet_name).strip()

    if sheet_name not in wb.sheetnames:
        print("[Sheet不存在]", sheet_name)
        sheet_not_found += 1
        total_unmatch += len(group)
        continue

    ws = wb[sheet_name]
    sheet_match += 1

    print("\n======================================")
    print("Sheet:", sheet_name)
    print("预测数据:", len(group))

    # --------------------------------------------------------
    # 6. 找到模板表头
    # --------------------------------------------------------
    header_row = None
    header = {}

    for r in range(1, min(ws.max_row, 30) + 1):

        row_values = [
            str(ws.cell(r, c).value).strip()
            if ws.cell(r, c).value is not None else ""
            for c in range(1, ws.max_column + 1)
        ]

        if "Biortus No." in row_values:
            header_row = r
            for c, value in enumerate(row_values, start=1):
                if value:
                    header[value] = c
            break

    if header_row is None:
        print("[错误] 没找到 Biortus No.")
        total_unmatch += len(group)
        continue

    # --------------------------------------------------------
    # 7. 找到原有两列
    # --------------------------------------------------------
    if "R2 Predicted Value" not in header or "R2 Predicted Rank" not in header:
        print("[错误] 没找到原有 R2 预测列")
        total_unmatch += len(group)
        continue

    old_pred_col = header["R2 Predicted Value"]
    old_rank_col = header["R2 Predicted Rank"]

    if old_rank_col != old_pred_col + 1:
        print("[错误] 原有两列不相邻")
        total_unmatch += len(group)
        continue

    # --------------------------------------------------------
    # 8. 在原有两列后新增两列
    # --------------------------------------------------------
    new_pred_col = old_rank_col + 1
    new_rank_col = old_rank_col + 2

    ws.insert_cols(new_pred_col, 2)

    ws.cell(header_row, new_pred_col).value = "R2 Predicted Value"
    ws.cell(header_row, new_rank_col).value = "R2 Predicted Rank"

    # --------------------------------------------------------
    # 9. 建立模板 Biortus No. → Excel 行号
    # --------------------------------------------------------
    excel_id_to_row = {}

    for r in range(header_row + 1, ws.max_row + 1):

        biortus_id = clean_id(ws.cell(r, header["Biortus No."]).value)

        if biortus_id:
            excel_id_to_row[biortus_id] = r

    # --------------------------------------------------------
    # 10. 预测 CSV Name → 模板 Biortus No. 匹配
    # --------------------------------------------------------
    match_count = 0

    for _, pred_row in group.iterrows():

        # 预测结果 Name 对应模板 Biortus No.
        biortus_id = pred_row["Name_match"]

        if not biortus_id:
            continue

        if biortus_id not in excel_id_to_row:
            total_unmatch += 1
            continue

        excel_row = excel_id_to_row[biortus_id]

        # 新增预测值
        ws.cell(excel_row, new_pred_col).value = round(
            float(pred_row["pred_Conv"]), 3
        )

        # 新增排名
        ws.cell(excel_row, new_rank_col).value = int(pred_row["rank"])

        match_count += 1
        total_match += 1

    print("成功匹配:", match_count)


# ============================================================
# 11. 保存
# ============================================================
wb.save(output_xlsx)

print("\n========== 处理完成 ==========")
print("Sheet匹配数量:", sheet_match)
print("不存在的Sheet:", sheet_not_found)
print("成功填入:", total_match)
print("未匹配:", total_unmatch)
print("输出文件:", output_xlsx)
