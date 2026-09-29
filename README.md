# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import os
import re
from tqdm import tqdm

def sanitize_filename(name):
    """
    清洗字符串，使其可以作为合法文件名
    """
    if pd.isna(name):
        return "unnamed"
    # 替换 Windows/Linux 不允许的字符
    clean_name = re.sub(r'[\\/:*?"<>|]', '_', str(name))
    return clean_name.strip()

def split_by_sm_and_rank(input_csv,smile_csv,output_dir, ascending=False):
    df = pd.read_csv(input_csv)
    smile_df = pd.read_csv(smile_csv)
    # 建立 SM -> Name 映射
    sm_to_name = dict(zip(smile_df['SM'], smile_df['Name']))

    # 1. 读取预测结果
    if not os.path.exists(input_csv):
        print(f"[!] 错误: 找不到输入文件 {input_csv}")
        return
    
    print(f"[*] 正在读取预测结果: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 检查必要的列
    #required_cols = ['Biortus No', 'Name', 'SM', 'pred_Conv']
    required_cols = ['Name', 'SM', 'pred_Conv']
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] 错误: 文件中缺失列 '{col}'")
            return

    # 2. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[*] 已创建输出目录: {output_dir}")

    # 3. 按 SM 分组处理
    # 注意：如果 SM 相同但 Name 不同，这里会以 SM 为准分组，文件名取该组第一个 Name
    grouped = df.groupby('SM')
    unique_sm_count = len(grouped)
    print(f"[*] 检测到 {len(grouped)} 个唯一的 SM，正在进行划分和排名...")

    # 使用 enumerate 获得序号 i，防止文件名重复覆盖
    for i, (sm_value, group) in enumerate(tqdm(grouped, desc="生成文件中")):
        # a. 根据 pred_Conv 排序
        sorted_group = group.sort_values(by='pred_Conv', ascending=ascending).reset_index(drop=True)
        
        # b. 新增 rank 列
        sorted_group['rank'] = sorted_group.index + 1

        #按照Biortus No排序输出
        biortus_num = pd.to_numeric(sorted_group['Name'],errors='coerce') #Biortus No,Name
        sorted_group['_sort_type'] = biortus_num.notna().astype(int)
        # 数字本身
        sorted_group['_sort_num'] = biortus_num
        # 排序：
        # _sort_type = 0 → 非数字 → 最前面
        # _sort_type = 1 → 数字
        # _sort_num → 数字从小到大
        sorted_group = sorted_group.sort_values(
            by=['_sort_type', '_sort_num'],
            ascending=[True, True]
        ).reset_index(drop=True)

        # 删除临时排序列
        sorted_group = sorted_group.drop(
            columns=['_sort_type', '_sort_num']
        )
        #固定输出列顺序
        #sorted_group=sorted_group[['Biortus No','Name','SM','pred_Conv','rank']]
        sorted_group=sorted_group[['Name','SM','pred_Conv','rank']]

        # pred_Conv 保留3位小数
        sorted_group['pred_Conv'] = sorted_group['pred_Conv'].round(3)
       
        # 从 SMILE.csv 查找对应 Name
        raw_name = sm_to_name.get(sm_value, f"Unknown_{i+1}")
        safe_name = sanitize_filename(raw_name)
        filename = f"{safe_name}.csv"
        output_path = os.path.join(output_dir, filename)

        
        # d. 保存文件
        sorted_group.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*30)
    print(f"✅ 处理完成！实际生成文件数: {unique_sm_count}")
    print(f"   输入文件: {input_csv}")
    print(f"   生成文件数: {len(grouped)}")
    print(f"   保存目录: {os.path.abspath(output_dir)}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 SM 划分预测结果并按 pred_Conv 排名")
    
    # 参数设置
    parser.add_argument("-i", "--input", required=True, help="输入的预测结果 CSV 文件")
    parser.add_argument("-o", "--output", required=True, help="输出文件夹路径")
    parser.add_argument("--reverse", action="store_true", help="如果设置，则 pred_Conv 越小排名越靠前 (默认越大越靠前)")
    parser.add_argument("-s","--smile",required=True,help="SMILE.csv文件路径")
    args = parser.parse_args()

    # 执行逻辑
    # 默认情况下 reverse=False, ascending=False (即从大到小排)
    # 如果设置了 --reverse, ascending=True (即从小到大排)
    split_by_sm_and_rank(args.input, args.smile, args.output, ascending=args.reverse)
