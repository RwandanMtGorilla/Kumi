"""
Markdown 到 QA 数据集处理流程 - 统合脚本

这个脚本整合了从 Markdown 文件到最终 QA 数据集的完整处理流程:
1. Step 1: Markdown 文件切分并转换为 CSV (step1_md2csv)
2. Step 2: 从 CSV 文本生成 QA 对 (step2_chunk2qa)
3. Step 3: 合并多个 CSV 文件为单个数据集 (step3_merge)

使用说明:
- 在 __main__ 中调整 base_folder 和其他参数
- 可以根据需要选择运行部分步骤或全部步骤
- 每个步骤的函数都可以独立调用
"""

import os
from step1_md2csv import process_md_to_csv
from step2_chunk2qa import process_csv_to_qa
from step3_merge import merge_csv_files


def run_step1(input_folder, output_folder, chunk_size=350, min_chunk_size=100):
    """
    Step 1: 将 Markdown/文本文件处理为 CSV

    参数:
        input_folder: Markdown 文件输入路径
        output_folder: CSV 文件输出路径
        chunk_size: 文本切分大小
        min_chunk_size: 最小块大小

    输出:
        CSV 文件,包含 Text, Text_pure, Img_url, Position 列
    """
    print("\n" + "="*60)
    print("Step 1: 处理 Markdown 文件为 CSV")
    print("="*60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"切分大小: {chunk_size}, 最小块: {min_chunk_size}")

    process_md_to_csv(
        input_folder=input_folder,
        output_folder=output_folder,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size
    )

    print("\nStep 1 完成!\n")


def run_step2(input_folder, output_folder, rounds=3, max_workers=5):
    """
    Step 2: 从 CSV 文本生成 QA 对

    参数:
        input_folder: CSV 文件输入路径
        output_folder: 带 QA 的 CSV 输出路径
        rounds: 每行生成轮次(选择最长结果)
        max_workers: 并发线程数

    输入:
        需要包含 Text_pure 列的 CSV 文件

    输出:
        CSV 文件,增加 Question 和 Answer 列
    """
    print("\n" + "="*60)
    print("Step 2: 生成 QA 对")
    print("="*60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"生成轮次: {rounds}, 最大并发: {max_workers}")

    process_csv_to_qa(
        input_folder=input_folder,
        output_folder=output_folder,
        rounds=rounds,
        max_workers=max_workers
    )

    print("\nStep 2 完成!\n")


def run_step3(input_folder, output_folder, output_filename):
    """
    Step 3: 合并多个 CSV 文件为单个数据集

    参数:
        input_folder: 多个 CSV 文件的输入路径
        output_folder: 合并后文件的输出路径
        output_filename: 合并后的文件名

    输出:
        单个合并的 CSV 文件,包含 file_name 列标记来源
    """
    print("\n" + "="*60)
    print("Step 3: 合并 CSV 文件")
    print("="*60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"输出文件名: {output_filename}")

    merge_csv_files(
        input_folder=input_folder,
        output_folder=output_folder,
        output_filename=output_filename
    )

    print("\nStep 3 完成!\n")


if __name__ == "__main__":
    # ===== 配置区域 =====

    # 数据集基础配置
    DATA_NAME = "notebook_lm"
    BASE_FOLDER = DATA_NAME  # 输入/输出的子文件夹名
    DATASET_NAME = DATA_NAME # 最终数据集名称

    # Step 1 参数
    CHUNK_SIZE = 350  # 文本切分大小
    MIN_CHUNK_SIZE = 100  # 最小块大小

    # Step 2 参数
    QA_ROUNDS = 3  # QA 生成轮次
    MAX_WORKERS = 5  # 并发线程数

    # ===== 选择运行模式 =====
    CURRENT_INPUT_FOLDER = "input"
    CURRENT_OUTPUT_FOLDER = "output"
    run_step1(
        input_folder=f"./{CURRENT_INPUT_FOLDER}/{BASE_FOLDER}/",
        output_folder=f"./{CURRENT_OUTPUT_FOLDER}/{BASE_FOLDER}/",
        chunk_size=CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE
    )

    # CURRENT_INPUT_FOLDER = CURRENT_OUTPUT_FOLDER
    # CURRENT_OUTPUT_FOLDER = "QA"
    # run_step2(
    #     input_folder=f"./{CURRENT_INPUT_FOLDER}/{BASE_FOLDER}/",
    #     output_folder=f"./{CURRENT_OUTPUT_FOLDER}/{BASE_FOLDER}/",
    #     rounds=QA_ROUNDS,
    #     max_workers=MAX_WORKERS
    # )

    CURRENT_INPUT_FOLDER = CURRENT_OUTPUT_FOLDER
    CURRENT_OUTPUT_FOLDER = "merged"

    run_step3(
        input_folder=f"./{CURRENT_INPUT_FOLDER}/{BASE_FOLDER}/",
        output_folder=f"./{CURRENT_OUTPUT_FOLDER}/",
        output_filename=f"{DATASET_NAME}.csv"
    )
