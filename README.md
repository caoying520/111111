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
