import os
import pandas as pd


# CSV 所在文件夹
input_dir = r"D:\你的CSV文件夹"

# 最终 Excel
output_excel = r"D:\你的CSV文件夹\总结果.xlsx"


# 获取所有 CSV 文件
csv_files = [
    f for f in os.listdir(input_dir)
    if f.lower().endswith(".csv")
]

# 按文件名排序
csv_files.sort()

print(f"[*] 找到 {len(csv_files)} 个 CSV 文件")


# 创建 Excel
with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    for i, csv_file in enumerate(csv_files):

        csv_path = os.path.join(input_dir, csv_file)

        print(f"[*] 正在处理: {csv_file}")

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # Sheet 名称
        # Excel Sheet 最多 31 个字符
        sheet_name = os.path.splitext(csv_file)[0][:31]

        # 写入 Excel
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )


print("\n" + "=" * 40)
print("✅ 转换完成！")
print(f"Excel 文件: {output_excel}")
print(f"Sheet 数量: {len(csv_files)}")
print("=" * 40)
