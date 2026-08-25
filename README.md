import os
import argparse
import pandas as pd


def csv_to_excel(input_dir, output_excel):

    # 检查输入目录
    if not os.path.isdir(input_dir):
        print(f"[!] 错误：找不到输入目录: {input_dir}")
        return

    # 获取所有 CSV
    csv_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]

    # 按文件名排序
    csv_files.sort()

    if not csv_files:
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

            # Sheet 名称使用 CSV 文件名
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


def main():

    parser = argparse.ArgumentParser(
        description="将多个 CSV 文件合并到一个 Excel 文件，每个 CSV 对应一个 Sheet"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV 文件所在的文件夹"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出 Excel 文件路径，例如 result.xlsx"
    )

    args = parser.parse_args()

    csv_to_excel(
        input_dir=args.input,
        output_excel=args.output
    )


if __name__ == "__main__":
    main()
