import os
from tqdm import tqdm
from markitdown import MarkItDown


def convert_files(input_folder, output_folder, file_extension):
    # 确保output文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取input文件夹中指定后缀名的所有文件
    files_to_convert = [f for f in os.listdir(input_folder) if f.endswith(file_extension)]

    # 实例化MarkItDown对象
    md = MarkItDown(enable_plugins=False)

    # 使用tqdm显示进度条
    for file in tqdm(files_to_convert, desc="Converting files"):
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file.replace(file_extension, '.md'))  # 假设输出为markdown格式

        # 读取文件并进行转换
        with open(input_file_path, 'rb') as f:
            result = md.convert(f)

        # 保存转换后的内容到输出文件夹
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(result.text_content)


# 调用函数，假设需要转换的文件是xlsx格式
convert_files('./resources/xlsx_input', './resources/xlsx_output', '.xlsx')
