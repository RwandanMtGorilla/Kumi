import os
import pandas as pd
from tqdm import tqdm

# 定义输入和输出文件夹
input_folder = "./output/GT_ML_TEST/"
output_folder = "./merged/"
output_filename = "GT_ML_TEST.csv"

# 创建输出文件夹(如果不存在)
os.makedirs(output_folder, exist_ok=True)


def get_unique_column_name(existing_columns, base_name="file_name"):
    """
    生成唯一的列名,如果列名已存在则添加序号

    参数:
        existing_columns: 现有列名列表
        base_name: 基础列名

    返回:
        唯一的列名
    """
    if base_name not in existing_columns:
        return base_name

    counter = 1
    while True:
        new_name = f"{base_name}{counter}"
        if new_name not in existing_columns:
            return new_name
        counter += 1


def merge_csv_files(input_folder, output_folder, output_filename):
    """
    合并文件夹中所有CSV文件

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        output_filename: 输出文件名

    逻辑:
        - 读取所有.csv文件
        - 保留所有不同的列名
        - 添加file_name列记录来源文件
        - 输出单个合并后的CSV文件
    """
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print(f"警告: 在文件夹 {input_folder} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 用于存储所有数据框
    all_dataframes = []

    # 逐个读取CSV文件
    for filename in tqdm(csv_files, desc="读取CSV文件"):
        try:
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path, encoding='utf-8-sig')

            # 记录原始文件名(不含后缀)
            source_file_name = os.path.splitext(filename)[0]

            # 暂时存储文件名信息
            df['__temp_source_file__'] = source_file_name

            all_dataframes.append(df)
            print(f"  读取文件: {filename} ({len(df)} 行, {len(df.columns)} 列)")

        except Exception as e:
            print(f"  错误: 读取文件 {filename} 时出错: {e}")
            continue

    if not all_dataframes:
        print("错误: 没有成功读取任何CSV文件")
        return

    # 合并所有数据框,保留所有列
    print("\n开始合并CSV文件...")
    merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    # 获取所有现有列名(不包括临时列)
    existing_columns = [col for col in merged_df.columns if col != '__temp_source_file__']

    # 生成唯一的file_name列名
    file_name_column = get_unique_column_name(existing_columns)

    # 将临时列重命名为最终列名
    merged_df.rename(columns={'__temp_source_file__': file_name_column}, inplace=True)

    # 将file_name列移到第一列
    cols = merged_df.columns.tolist()
    cols.remove(file_name_column)
    cols = [file_name_column] + cols
    merged_df = merged_df[cols]

    # 输出合并结果信息
    print(f"\n合并完成:")
    print(f"  总行数: {len(merged_df)}")
    print(f"  总列数: {len(merged_df.columns)}")
    print(f"  文件名列: {file_name_column}")
    print(f"  所有列名: {list(merged_df.columns)}")

    # 保存合并后的文件
    output_filepath = os.path.join(output_folder, output_filename)
    merged_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

    print(f"\n合并文件已保存: {output_filepath}")


if __name__ == "__main__":
    # 执行合并操作
    merge_csv_files(input_folder, output_folder, output_filename)
