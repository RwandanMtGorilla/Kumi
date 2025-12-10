import pandas as pd
import re
import os
import sys
from tqdm import tqdm
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径,以便导入 config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings


# 正则表达式模式,用于解析 QA 对
pattern = re.compile(r"-?\s*question: (.*?)\s*-?\s*answer: (.+)", re.DOTALL)


def generate(prompt, model_name=None, system=None, temperature=None,
             max_tokens=None, stream=False, callback=None, api_key=None, base_url=None):
    """
    使用 OpenAI API 格式生成响应

    参数:
        prompt: 用户输入的提示词
        model_name: 模型名称(可选,默认从环境变量读取)
        system: 系统提示词(可选)
        temperature: 温度参数(可选,0-2之间)
        max_tokens: 最大token数(可选)
        stream: 是否流式输出(默认False)
        callback: 回调函数(可选)
        api_key: API密钥(可选,默认从配置读取)
        base_url: API基础URL(可选,默认从配置读取)

    返回:
        full_response: 完整的响应文本
        usage_info: token使用信息
    """
    try:
        # 从配置读取或使用参数
        api_key = api_key or settings.OPENAI_API_KEY
        base_url = base_url or settings.OPENAI_BASE_URL
        default_model = "qwen/qwen3-vl-235b-a22b-instruct"

        url = f"{base_url}/chat/completions"

        # 构建消息列表
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 构建请求payload
        payload = {
            "model": model_name or default_model,
            "messages": messages,
            "stream": stream
        }

        # 添加可选参数
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        with requests.post(url, json=payload, headers=headers, stream=stream) as response:
            response.raise_for_status()

            full_response = ""
            usage_info = None

            if stream:
                # 流式响应处理
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')

                        # 跳过注释行
                        if line.startswith(':'):
                            continue

                        # 移除 "data: " 前缀
                        if line.startswith('data: '):
                            line = line[6:]

                        # 检查是否结束
                        if line == '[DONE]':
                            break

                        try:
                            chunk = json.loads(line)

                            # 如果提供了回调函数,调用它
                            if callback:
                                callback(chunk)
                            else:
                                # 提取内容
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        print(content, end="", flush=True)

                            # 获取使用信息(通常在最后一个chunk)
                            if 'usage' in chunk:
                                usage_info = chunk['usage']

                        except json.JSONDecodeError:
                            continue

                if not callback:
                    print()  # 换行
            else:
                # 非流式响应处理
                result = response.json()

                if callback:
                    callback(result)
                else:
                    if 'choices' in result and len(result['choices']) > 0:
                        full_response = result['choices'][0]['message']['content']
                        print(full_response)

                usage_info = result.get('usage')

            return full_response, usage_info

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None, None
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None


def process_single_row(index, row, file_name, rounds=3):
    """
    处理单行数据，生成QA对

    参数:
        index: 行索引
        row: DataFrame的行数据
        file_name: 文件名（用于日志）
        rounds: 生成轮次(默认3轮,选择最长结果)

    返回:
        tuple: (index, row_data) - 索引和包含Question/Answer的行数据字典
    """
    from prompts import SYS_ED_TEMPLATE, ED_TEMPLATE

    text = row['Text_pure']
    if not isinstance(text, str):
        print(f"Skipping row {index} in file {file_name}: Text is not a string")
        row_data = row.to_dict()
        row_data['Question'] = ''
        row_data['Answer'] = ''
        return index, row_data

    best_question = ''
    best_answer = ''
    best_length = 0

    # 执行多轮生成，选择最长的结果
    for _ in range(rounds):
        sys_prompt = SYS_ED_TEMPLATE
        prompt = ED_TEMPLATE.format(text=text)

        response, _ = generate(
            system=sys_prompt,
            prompt=prompt,
            temperature=0.7
        )

        if response:
            match = pattern.search(response)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                length = len(question) + len(answer)
                if length > best_length:
                    best_question = question
                    best_answer = answer
                    best_length = length

    # 构建返回数据
    row_data = row.to_dict()
    row_data['Question'] = best_question
    row_data['Answer'] = best_answer

    return index, row_data


def process_csv_to_qa(input_folder, output_folder, rounds=3, max_workers=5):
    """
    将 CSV 文件中的文本转换为 QA 对

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        rounds: 每行生成的轮次(默认3,选择最长结果)
        max_workers: 最大并发线程数(默认5)

    逻辑:
        - 读取所有 .csv 文件
        - 对每行的 Text_pure 列生成 Question 和 Answer
        - 使用多线程并发处理
        - 支持断点续传(跳过已处理的行)
        - 输出包含 Question 和 Answer 列的 CSV
    """
    # 创建输出文件夹(如果不存在)
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_name in tqdm(csv_files, desc="Processing files"):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # 加载已有结果（如果存在）
        if os.path.exists(output_file_path):
            new_df = pd.read_csv(output_file_path, encoding='utf-8-sig')
        else:
            new_df = pd.DataFrame(columns=['Question', 'Answer'])

        # 读取输入数据
        df = pd.read_csv(input_file_path, encoding='utf-8-sig')

        # 识别需要处理的行（跳过已完成的行）
        rows_to_process = []
        for index, row in df.iterrows():
            # 检查是否已经处理过
            if index < len(new_df) and pd.notnull(new_df.loc[index, 'Question']) and pd.notnull(new_df.loc[index, 'Answer']):
                continue  # 跳过已完成的行
            rows_to_process.append((index, row))

        if not rows_to_process:
            print(f"File {file_name}: All rows already processed.")
            continue

        print(f"Processing {len(rows_to_process)} rows in {file_name} with {max_workers} workers...")

        # 使用线程池并发处理
        results = {}  # 使用字典存储结果，key为index
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_single_row, index, row, file_name, rounds): index
                for index, row in rows_to_process
            }

            # 使用tqdm显示进度，并收集结果
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Processing {file_name}"):
                try:
                    index, row_data = future.result()
                    results[index] = row_data
                except Exception as e:
                    index = future_to_index[future]
                    print(f"\nError processing row {index}: {e}")
                    # 发生错误时，创建空结果
                    results[index] = df.loc[index].to_dict()
                    results[index]['Question'] = ''
                    results[index]['Answer'] = ''

        # 按索引顺序更新DataFrame
        for index in sorted(results.keys()):
            row_data = results[index]
            if index < len(new_df):
                new_df.iloc[index] = row_data
            else:
                new_df = pd.concat([new_df, pd.DataFrame([row_data])], ignore_index=True)

        # 保存最终结果
        new_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"File {file_name} processed and saved to {output_file_path}.")


if __name__ == "__main__":
    # 定义文件夹路径
    input_folder = 'output'
    output_folder = 'QA'

    # 执行处理
    process_csv_to_qa(input_folder, output_folder, rounds=3, max_workers=5)
