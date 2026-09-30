import pandas as pd
from openpyxl import load_workbook

# ============================================================
# 文件路径
# ============================================================

# 预测结果
pred_csv = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/model_Activity_finetune_cls_2/predictions.csv"

# SM 和 Sheet名称对应关系
smile_csv = "/data/caoying/softwares/酶R2/enzyme_交接20260609/微调/SMILE.csv"

# Excel模板
template_xlsx = "/你的Excel模板.xlsx"

# 输出Excel
output_xlsx = "/你的Excel模板_填充结果.xlsx"


# ============================================================
# 清洗 Biortus 编号
# ============================================================
def clean_id(x):

    if pd.isna(x):
        return ""

    if isinstance(x, float) and x.is_integer():
        return str(int(x))

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


# ============================================================
# 清洗 SM
# ============================================================
def clean_sm(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# 1. 读取预测结果
# ============================================================
pred_df = pd.read_csv(pred_csv)

print("预测数据数量:", len(pred_df))
print("预测数据列:", pred_df.columns.tolist())


pred_df["SM_match"] = pred_df["SM"].apply(clean_sm)
pred_df["Name_match"] = pred_df["Name"].apply(clean_id)


# ============================================================
# 2. 读取 SMILE.csv
#
# SM → Name
#
# Name 就是 Excel Sheet 名称
# ============================================================
smile_df = pd.read_csv(smile_csv)

print("\nSMILE.csv 列:", smile_df.columns.tolist())
print("SMILE.csv 数量:", len(smile_df))


smile_df["SM_match"] = smile_df["SM"].apply(clean_sm)
smile_df["Sheet_Name"] = smile_df["Name"].astype(str).str.strip()


# 建立：
#
# SM → Sheet名称
#
sm_to_sheet = dict(
    zip(
        smile_df["SM_match"],
        smile_df["Sheet_Name"]
    )
)


# ============================================================
# 3. 根据预测结果中的 SM
#    找到 Excel Sheet 名称
# ============================================================
pred_df["Sheet_Name"] = pred_df["SM_match"].map(sm_to_sheet)


# 检查哪些 SM 没找到
missing_sm = pred_df[
    pred_df["Sheet_Name"].isna()
]["SM_match"].unique()


if len(missing_sm) > 0:

    print("\n[警告] 以下 SM 在 SMILE.csv 中没有找到对应 Sheet：")

    for sm in missing_sm[:20]:
        print(" ", sm)

    if len(missing_sm) > 20:
        print(" ... 共", len(missing_sm), "个")


# ============================================================
# 4. 每个 SM 内部计算 rank
#
# pred_Conv 越大，rank 越靠前
# ============================================================
pred_df["rank"] = (
    pred_df.groupby("SM_match")["pred_Conv"]
    .rank(
        method="min",
        ascending=False
    )
    .astype("Int64")
)


# ============================================================
# 5. 打开 Excel
# ============================================================
wb = load_workbook(template_xlsx)

print("\nExcel Sheet 数量:", len(wb.sheetnames))


total_match = 0
total_unmatch = 0
sheet_match = 0
sheet_not_found = 0


# ============================================================
# 6. 按 Sheet 处理
# ============================================================
for sheet_name, group in pred_df.groupby(
    "Sheet_Name",
    dropna=True
):

    sheet_name = str(sheet_name).strip()


    # --------------------------------------------------------
    # Excel 中找 Sheet
    # --------------------------------------------------------
    if sheet_name not in wb.sheetnames:

        print(
            f"\n[Sheet不存在] "
            f"Sheet = {sheet_name}"
        )

        sheet_not_found += 1
        total_unmatch += len(group)

        continue


    ws = wb[sheet_name]

    sheet_match += 1


    print("\n======================================")
    print("SM:", group["SM_match"].iloc[0])
    print("Sheet:", sheet_name)
    print("预测数据:", len(group))


    # ========================================================
    # 7. 找表头
    # ========================================================
    header_row = None
    header = {}


    for r in range(
        1,
        min(ws.max_row, 30) + 1
    ):

        row_values = []

        for c in range(
            1,
            ws.max_column + 1
        ):

            value = ws.cell(r, c).value

            if value is None:
                value = ""

            value = str(value).strip()

            row_values.append(value)


        if "Biortus No." in row_values:

            header_row = r

            for c, value in enumerate(
                row_values,
                start=1
            ):

                if value:
                    header[value] = c

            break


    if header_row is None:

        print(
            "[错误] 没找到 Biortus No. 表头"
        )

        total_unmatch += len(group)

        continue


    # ========================================================
    # 8. 找/创建预测值列
    # ========================================================
    if "R2 Predicted Value" not in header:

        new_col = ws.max_column + 1

        ws.cell(
            header_row,
            new_col
        ).value = "R2 Predicted Value"

        header["R2 Predicted Value"] = new_col


    # ========================================================
    # 9. 找/创建排名列
    # ========================================================
    if "R2 Predicted Rank" not in header:

        new_col = ws.max_column + 1

        ws.cell(
            header_row,
            new_col
        ).value = "R2 Predicted Rank"

        header["R2 Predicted Rank"] = new_col


    biortus_col = header["Biortus No."]
    pred_col = header["R2 Predicted Value"]
    rank_col = header["R2 Predicted Rank"]


    # ========================================================
    # 10. 建立：
    #
    # Excel Biortus No. → Excel 行号
    # ========================================================
    excel_id_to_row = {}


    for r in range(
        header_row + 1,
        ws.max_row + 1
    ):

        biortus_id = clean_id(
            ws.cell(
                r,
                biortus_col
            ).value
        )

        if biortus_id:

            excel_id_to_row[
                biortus_id
            ] = r


    # ========================================================
    # 11. 根据 Name → Biortus No.
    #     填入预测和排名
    # ========================================================
    match_count = 0


    for _, pred_row in group.iterrows():

        biortus_id = pred_row["Name_match"]


        if biortus_id == "":
            continue


        # ----------------------------------------------
        # Excel中寻找对应 Biortus No.
        # ----------------------------------------------
        if biortus_id not in excel_id_to_row:

            total_unmatch += 1

            continue


        excel_row = excel_id_to_row[
            biortus_id
        ]


        # ----------------------------------------------
        # 预测值
        # ----------------------------------------------
        ws.cell(
            excel_row,
            pred_col
        ).value = round(
            float(pred_row["pred_Conv"]),
            3
        )


        # ----------------------------------------------
        # Rank
        # ----------------------------------------------
        ws.cell(
            excel_row,
            rank_col
        ).value = int(
            pred_row["rank"]
        )


        match_count += 1
        total_match += 1


    print(
        "成功填入:",
        match_count
    )


# ============================================================
# 12. 保存
# ============================================================
wb.save(output_xlsx)


# ============================================================
# 13. 最终统计
# ============================================================
print("\n")
print("======================================")
print("处理完成")
print("======================================")

print(
    "SM → Sheet 匹配数量:",
    sheet_match
)

print(
    "不存在的 Sheet:",
    sheet_not_found
)

print(
    "成功填入:",
    total_match
)

print(
    "未匹配:",
    total_unmatch
)

print(
    "输出文件:",
    output_xlsx
)
