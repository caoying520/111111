def split_by_sm_and_rank(input_csv, output_dir, ascending=False):
    # 1. 读取预测结果
    if not os.path.exists(input_csv):
        print(f"[!] 错误: 找不到输入文件 {input_csv}")
        return
    
    print(f"[*] 正在读取预测结果: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 检查必要的列
    required_cols = ['Biortus No', 'Name', 'SM', 'pred_Conv']
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
        
        # c. 确定文件名：加上序号 i，彻底解决同名覆盖问题
        raw_name = sorted_group['Name'].iloc[0]
        safe_name = sanitize_filename(raw_name)
        
        # 文件名格式：序号_名称_ranked.csv
        filename = f"{i+1:03d}_{safe_name}_ranked.csv" 
        output_path = os.path.join(output_dir, filename)



        
        # d. 保存文件
        sorted_group.to_csv(output_path, index=False, encoding='utf-8-sig')



    print("\n" + "="*30)
    print(f"✅ 处理完成！实际生成文件数: {unique_sm_count}")
    print(f"   输入文件: {input_csv}")
    print(f"   生成文件数: {len(grouped)}")
    print(f"   保存目录: {os.path.abspath(output_dir)}")
    print("="*30)    按照Biortus No,Name,SM,pred_Conv,rank
排序输出
