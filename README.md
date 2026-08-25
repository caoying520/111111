import os
import sys
import pandas as pd


def csv_to_excel(input_dir, output_excel):

    # 检查输入目录
    if not os.path.isdir(input_dir):
        print(f"[!] 错误：找不到输入目录: {input_dir}")
        return

    # 获取 CSV 文件
    csv_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]

    # 按文件名排序
    csv_files.sort()

    if len(csv_files) == 0:
        print(f"[!] 输入目录中没有找到 CSV 文件: {input_dir}")
        return

    print(f"[*] 找到 {len(csv_files)} 个 CSV 文件")

    # 创建 Excel
    with pd.ExcelWriter(
        output_excel,
        engine="openpyxl"
    ) as writer:

        for i, csv_file in enumerate(csv_files, start=1):

            csv_path = os.path.join(input_dir, csv_file)

            print(f"[*] [{i}/{len(csv_files)}] 正在处理: {csv_file}")

            # 读取 CSV
            df = pd.read_csv(csv_path)

            # Excel Sheet 名称
            sheet_name = os.path.splitext(csv_file)[0]

            # Excel Sheet 名最长 31 个字符
            sheet_name = sheet_name[:31]

            # 写入 Excel
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print("\n" + "=" * 40)
    print("✅ 转换完成！")
    print(f"CSV 数量: {len(csv_files)}")
    print(f"Excel 文件: {os.path.abspath(output_excel)}")
    print("=" * 40)


if __name__ == "__main__":

    # 检查命令行参数
    if len(sys.argv) != 3:
        print(
            "用法:\n"
            "python csv_to_excel.py <CSV文件夹> <输出Excel文件>\n\n"
            "示例:\n"
            'python csv_to_excel.py "D:\\result" "D:\\result\\总结果.xlsx"'
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_excel = sys.argv[2]

    csv_to_excel(input_dir, output_excel)
