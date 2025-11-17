import pandas as pd
import re
import os
from tqdm import tqdm
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import settings
# 加载环境变量

# 从环境变量读取配置
API_KEY = settings.OPENAI_API_KEY
BASE_URL = settings.OPENAI_BASE_URL
DEFAULT_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# todo: 使用json格式输出 requests里有参数可设置固定输出格式为json
pattern = re.compile(r"-?\s*question: (.*?)\s*-?\s*answer: (.+)", re.DOTALL)

# 定义文件夹路径
input_folder = 'output'
output_folder = 'QA'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rounds = 3  # 轮次
MAX_WORKERS = 5  # 最大并发线程数

def generate(prompt, model_name=None, system=None, temperature=None, 
             max_tokens=None, stream=False, callback=None):
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
    
    返回:
        full_response: 完整的响应文本
        usage_info: token使用信息
    """
    try:
        url = f"{BASE_URL}/chat/completions"
        
        # 构建消息列表
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # 构建请求payload
        payload = {
            "model": model_name or DEFAULT_MODEL,
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
            "Authorization": f"Bearer {API_KEY}",
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


def process_single_row(index, row, file_name):
    """
    处理单行数据，生成QA对

    参数:
        index: 行索引
        row: DataFrame的行数据
        file_name: 文件名（用于日志）

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


for file_name in tqdm(os.listdir(input_folder), desc="Processing files"):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # 加载已有结果（如果存在）
        if os.path.exists(output_file_path):
            new_df = pd.read_csv(output_file_path)
        else:
            new_df = pd.DataFrame(columns=['Question', 'Answer'])

        # 读取输入数据
        df = pd.read_csv(input_file_path)

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

        print(f"Processing {len(rows_to_process)} rows in {file_name} with {MAX_WORKERS} workers...")

        # 使用线程池并发处理
        results = {}  # 使用字典存储结果，key为index
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_single_row, index, row, file_name): index
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
