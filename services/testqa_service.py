import yaml
import re
from llm.factory import LLMFactory
import json
import pandas as pd
import requests
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
from config.settings import settings


class ModelEvaluator:
    def __init__(self,
                 yaml_folder=None,
                 output_folder=None):

        # 使用配置文件中的绝对路径，如果传入参数则使用参数
        self.yaml_folder = yaml_folder or settings.YAML_RULES_PATH
        self.output_folder = output_folder or settings.EVALUATION_RESULTS_PATH

        # 确保输出文件夹存在
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # 打印当前使用的路径
        print(f"📁 YAML规则文件夹路径: {self.yaml_folder}")
        print(f"📁 输出文件夹路径: {self.output_folder}")

        # 创建LLM客户端
        try:
            self.llm_client = LLMFactory.create_client()
        except Exception as e:
            print(f"❌ 创建LLM客户端失败: {e}")
            self.llm_client = None

        self.lock = Lock()

    def _normalize_text(self, text):
        """处理文本中的换行符和特殊字符"""
        if not isinstance(text, str):
            text = str(text)

        # 处理常见的换行符编码
        text = text.replace('\\n', '\n')  # 处理字面意义的 \n
        text = text.replace('\\r', '\r')  # 处理字面意义的 \r
        text = text.replace('\\t', '\t')  # 处理字面意义的 \t

        # 处理其他可能的转义字符
        text = text.replace('\\"', '"')  # 处理转义的双引号
        text = text.replace("\\'", "'")  # 处理转义的单引号

        return text

    def _load_yaml_rule(self, yaml_filename):
        """加载YAML评测规则"""
        yaml_path = os.path.join(self.yaml_folder, yaml_filename)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML文件 {yaml_path} 不存在")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件格式错误: {e}")

    def _fill_prompt_template(self, prompt_template, variables):
        """填充提示词模板"""
        filled_prompt = prompt_template

        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in filled_prompt:
                filled_prompt = filled_prompt.replace(placeholder, str(value))

        return filled_prompt

    def _extract_json_from_response(self, response_text):
        """从模型回答中提取JSON"""
        if not response_text:
            return {"score": 0.0, "reason": "模型无回应"}

        # 先规范化文本
        response_text = self._normalize_text(response_text)

        try:
            # 首先尝试直接解析
            result = json.loads(response_text.strip())
            # 对结果中的字符串字段进行规范化处理
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, str):
                        result[key] = self._normalize_text(value)
            return result
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON代码块
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                result = json.loads(matches[0])
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, str):
                            result[key] = self._normalize_text(value)
                return result
            except json.JSONDecodeError:
                pass

        # 尝试提取花括号内的内容
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, str):
                            result[key] = self._normalize_text(value)
                return result
            except json.JSONDecodeError:
                continue

        # 如果都失败了，尝试创建一个基本的JSON结构
        score_pattern = r'"?score"?\s*:\s*([0-9.]+)'
        reason_pattern = r'"?reason"?\s*:\s*"([^"]*)"'

        score_match = re.search(score_pattern, response_text)
        reason_match = re.search(reason_pattern, response_text)

        if score_match:
            result = {"score": float(score_match.group(1))}
            if reason_match:
                result["reason"] = self._normalize_text(reason_match.group(1))
            else:
                result["reason"] = "未找到具体原因"
            return result

        # 最后的备选方案
        return {
            "score": 0.0,
            "reason": self._normalize_text(f"无法解析评测结果: {response_text[:100]}...")
        }

    def list_yaml_rules(self):
        """列出所有可用的YAML规则文件"""
        if not os.path.exists(self.yaml_folder):
            Path(self.yaml_folder).mkdir(parents=True, exist_ok=True)
            return []

        yaml_files = [f for f in os.listdir(self.yaml_folder) if f.endswith('.yaml') or f.endswith('.yml')]
        return yaml_files

    def evaluate_results(self, result_file_path, yaml_filename, progress_callback=None, max_workers=3):
        """
        评测模型结果

        Args:
            result_file_path: 上一步生成的结果文件路径（绝对路径）
            yaml_filename: YAML评测规则文件名
            progress_callback: 进度回调函数，接收(current, total, message)参数

        Returns:
            str: 评测结果文件路径
        """
        if self.llm_client is None:
            raise RuntimeError("LLM客户端未初始化，无法进行评测")

        # 加载YAML规则
        rule_config = self._load_yaml_rule(yaml_filename)

        # 验证YAML规则格式
        if 'evaluation_rule' not in rule_config:
            raise ValueError("YAML文件中缺少 'evaluation_rule' 配置")

        if 'prompt' not in rule_config['evaluation_rule']:
            raise ValueError("YAML文件中缺少 'prompt' 配置")

        # 读取结果文件
        if not os.path.exists(result_file_path):
            raise FileNotFoundError(f"结果文件 {result_file_path} 不存在")

        try:
            if result_file_path.endswith('.xlsx'):
                df = pd.read_excel(result_file_path)
            elif result_file_path.endswith('.csv'):
                # 尝试不同的编码
                for encoding in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        df = pd.read_csv(result_file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"无法读取CSV文件，请检查编码格式")
            else:
                raise ValueError("不支持的文件格式，请使用 .xlsx 或 .csv 文件")
        except Exception as e:
            raise ValueError(f"读取结果文件失败: {e}")

        total_records = len(df)
        start_time = time.time()
        completed_count = 0
        evaluation_results = [None] * total_records  # 预分配结果列表

        print(f"🔍 开始评测 {total_records} 条记录 (并发数: {max_workers})")
        print(f"📋 使用规则: {rule_config['evaluation_rule']['name']}")
        print(f"📁 YAML文件: {os.path.join(self.yaml_folder, yaml_filename)}")
        print(f"📁 结果文件: {result_file_path}")
        print(f"📤 输出路径: {self.output_folder}")
        print("-" * 60)

        if progress_callback:
            progress_callback(0, total_records, "开始评测任务", 0)

            # 获取评测提示词模板
        prompt_template = rule_config['evaluation_rule']['prompt']['role']['description']

        def evaluate_single_record(args):
            """评测单条记录的函数"""
            index, row = args

            try:
                # 准备变量字典
                variables = {
                    'input': str(row.get('input', '')),
                    'generated_output': str(row.get('model_output', '')),
                    'reference_output': str(row.get('reference_output', '')),
                    'group': str(row.get('group', '')),
                }

                # 添加所有原始列作为可能的变量
                for col in df.columns:
                    if col not in variables:
                        variables[col] = str(row.get(col, ''))

                # 填充提示词
                filled_prompt = self._fill_prompt_template(prompt_template, variables)

                # 调用LLM进行评测
                evaluation_response = self.llm_client.simple_chat(
                    filled_prompt,
                    max_tokens=500,
                    temperature=rule_config.get('model_config', {}).get('temperature', 0.5)
                )

                # 解析评测结果
                evaluation_json = self._extract_json_from_response(evaluation_response)

                # 构建最终结果
                result_row = row.to_dict()
                result_row['evaluation_score'] = evaluation_json.get('score', 0.0)
                result_row['evaluation_reason'] = evaluation_json.get('reason', '无原因')
                result_row['evaluation_rule'] = rule_config['evaluation_rule']['name']
                result_row['evaluation_timestamp'] = datetime.now().isoformat()

                return index, result_row, None

            except Exception as e:
                error_msg = f"评测第 {index + 1} 条记录失败: {str(e)}"
                print(f"❌ {error_msg}")

                # 构建错误结果
                result_row = row.to_dict()
                result_row['evaluation_score'] = 0.0
                result_row['evaluation_reason'] = f"评测过程出错: {str(e)}"
                result_row['evaluation_rule'] = rule_config['evaluation_rule']['name']
                result_row['evaluation_timestamp'] = datetime.now().isoformat()

                return index, result_row, str(e)

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='evaluation_thread') as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(evaluate_single_record, (index, row)): index
                for index, row in df.iterrows()
            }

            # 处理完成的任务
            for future in as_completed(future_to_index):
                try:
                    index, result_row, error = future.result()
                    evaluation_results[index] = result_row

                    with self.lock:
                        completed_count += 1

                        # 计算预估剩余时间
                        elapsed_time = time.time() - start_time
                        if completed_count > 0:
                            avg_time_per_record = elapsed_time / completed_count
                            remaining_records = total_records - completed_count
                            estimated_remaining_time = avg_time_per_record * remaining_records
                        else:
                            estimated_remaining_time = 0

                        print(
                            f"✅ 评测完成第 {completed_count}/{total_records} 条 - 分数: {result_row.get('evaluation_score', 'N/A')}")

                        # 更新进度回调
                        if progress_callback:
                            progress_callback(
                                completed_count,
                                total_records,
                                f"已完成第 {completed_count}/{total_records} 条记录评测",
                                estimated_remaining_time
                            )

                except Exception as e:
                    print(f"❌ 处理评测结果时出错: {e}")

        # 保存最终评测结果
        final_results = [r for r in evaluation_results if r is not None]
        final_df = pd.DataFrame(final_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成输出文件名
        base_name = os.path.basename(result_file_path).split('.')[0]
        rule_name = rule_config['evaluation_rule']['name'].replace(' ', '_')
        safe_rule_name = "".join(c for c in rule_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        output_filename = f"evaluation_{base_name}_{safe_rule_name}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_folder, output_filename)

        final_df.to_excel(output_path, index=False, engine='openpyxl')

        # 最终完成回调
        if progress_callback:
            progress_callback(total_records, total_records, "评测完成，正在保存结果文件", 0)

        # 打印统计信息
        avg_score = final_df['evaluation_score'].mean()
        max_score = final_df['evaluation_score'].max()
        min_score = final_df['evaluation_score'].min()

        print(f"\n📈 评测完成统计:")
        print(f"   平均分数: {avg_score:.2f}")
        print(f"   最高分数: {max_score:.2f}")
        print(f"   最低分数: {min_score:.2f}")
        print(f"   总记录数: {len(final_df)}")
        print(f"   结果保存到: {output_path}")

        return output_path


class DifyWorkflowCaller:
    def __init__(self,
                 config_file=None,
                 table_folder=None,
                 output_folder=None):

        # 使用配置文件中的绝对路径，如果传入参数则使用参数
        self.config_file = config_file or settings.WORKFLOWS_CONFIG_PATH
        self.table_folder = table_folder or settings.TABLE_FOLDER_PATH
        self.output_folder = output_folder or settings.TEMP_RESULTS_PATH

        # 确保输出文件夹存在
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # 打印当前使用的路径
        print(f"📁 配置文件路径: {self.config_file}")
        print(f"📁 表格文件夹路径: {self.table_folder}")
        print(f"📁 输出文件夹路径: {self.output_folder}")

        # 加载工作流配置
        self.workflows = self._load_workflows()
        self.lock = Lock()

    def _call_dify_api_with_retry(self, workflow_config, query, inputs=None, max_retries=3):
        """带重试的API调用"""
        for attempt in range(max_retries):
            try:
                return self._call_dify_api(workflow_config, query, inputs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # 重试前等待1秒

    def _normalize_text(self, text):
        """处理文本中的换行符和特殊字符"""
        if not isinstance(text, str):
            text = str(text)

        # 处理常见的换行符编码
        text = text.replace('\\n', '\n')  # 处理字面意义的 \n
        text = text.replace('\\r', '\r')  # 处理字面意义的 \r
        text = text.replace('\\t', '\t')  # 处理字面意义的 \t

        # 处理其他可能的转义字符
        text = text.replace('\\"', '"')  # 处理转义的双引号
        text = text.replace("\\'", "'")  # 处理转义的单引号

        return text

    def _load_workflows(self):
        """加载工作流配置"""
        try:
            if not os.path.exists(self.config_file):
                print(f"❌ 配置文件 {self.config_file} 不存在")
                # 尝试创建配置文件的目录
                Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
                return {}

            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {wf['name']: wf for wf in config['workflows']}
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return {}

    def _call_dify_api(self, workflow_config, query, inputs=None):
        """调用Dify API"""
        headers = {
            "Authorization": f"Bearer {workflow_config['api_key']}",
            "Content-Type": "application/json"
        }

        data = {
            "inputs": inputs or {},  # 支持额外 inputs
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "abc-123"
        }

        try:
            response = requests.post(workflow_config['url'], headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API调用失败: {e}")
            return {"error": str(e)}

    def process_table(self, table_filename, workflow_name, progress_callback=None, max_workers=5):
        """
        处理表格文件，调用工作流API（支持并发）

        Args:
            table_filename: 表格文件名
            workflow_name: 工作流名称
            progress_callback: 进度回调函数，接收(current, total, message, estimated_time)参数
            max_workers: 最大并发数

        Returns:
            str: 输出文件路径
        """
        # 检查工作流是否存在
        if workflow_name not in self.workflows:
            available_workflows = list(self.workflows.keys())
            raise ValueError(f"工作流 '{workflow_name}' 不存在。可用工作流: {available_workflows}")

        workflow_config = self.workflows[workflow_name]

        # 构建表格文件的绝对路径
        table_path = os.path.join(self.table_folder, table_filename)
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"表格文件 {table_path} 不存在")

        # 根据文件扩展名选择读取方式
        if table_filename.endswith('.xlsx'):
            df = pd.read_excel(table_path)
        elif table_filename.endswith('.csv'):
            # 尝试不同的编码
            for encoding in ['utf-8', 'gbk', 'gb2312']:
                try:
                    df = pd.read_csv(table_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法读取CSV文件 {table_path}，请检查编码格式")
        else:
            raise ValueError("不支持的文件格式，请使用 .xlsx 或 .csv 文件")

        # 检查是否有input列
        if 'input' not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"表格文件中缺少 'input' 列。可用列: {available_columns}")

        total_records = len(df)
        start_time = time.time()
        completed_count = 0
        results = [None] * total_records  # 预分配结果列表

        print(f"🔍 开始处理 {total_records} 条记录 (并发数: {max_workers})")
        print(f"📊 使用工作流: {workflow_name}")
        print(f"📁 表格文件: {table_path}")
        print(f"📤 输出路径: {self.output_folder}")
        print("-" * 60)

        # 初始化进度回调
        if progress_callback:
            progress_callback(0, total_records, "开始处理数据集", 0)

        def process_single_record(args):
            """处理单条记录的函数"""
            index, row = args
            input_text = str(row['input'])

            # 构造 inputs（去掉 input 列）
            inputs = {col: row[col] for col in df.columns if col != 'input'}

            try:
                # 调用API
                api_response = self._call_dify_api_with_retry(workflow_config, input_text, inputs)

                # 提取回答内容
                if 'error' not in api_response:
                    answer = api_response.get('answer', '')
                    if not answer and 'data' in api_response:
                        answer = api_response['data'].get('answer', '')
                    if not answer and 'choices' in api_response:
                        answer = api_response['choices'][0].get('message', {}).get('content', '')
                    if not answer:
                        answer = str(api_response)

                    answer = self._normalize_text(answer)
                else:
                    answer = f"API调用失败: {api_response['error']}"

                # 构建结果记录
                result_row = row.to_dict()
                result_row['model_output'] = answer
                result_row['workflow_name'] = workflow_name
                result_row['timestamp'] = datetime.now().isoformat()

                return index, result_row, None

            except Exception as e:
                error_msg = f"处理第 {index + 1} 条记录失败: {str(e)}"
                print(f"❌ {error_msg}")

                # 构建错误结果
                result_row = row.to_dict()
                result_row['model_output'] = f"处理失败: {str(e)}"
                result_row['workflow_name'] = workflow_name
                result_row['timestamp'] = datetime.now().isoformat()

                return index, result_row, str(e)

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='dify_api_thread') as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_single_record, (index, row)): index
                for index, row in df.iterrows()
            }

            # 处理完成的任务
            for future in as_completed(future_to_index):
                try:
                    index, result_row, error = future.result()
                    results[index] = result_row

                    with self.lock:
                        completed_count += 1

                        # 计算预估剩余时间
                        elapsed_time = time.time() - start_time
                        if completed_count > 0:
                            avg_time_per_record = elapsed_time / completed_count
                            remaining_records = total_records - completed_count
                            estimated_remaining_time = avg_time_per_record * remaining_records
                        else:
                            estimated_remaining_time = 0

                        print(f"✅ 完成第 {completed_count}/{total_records} 条")

                        # 更新进度回调
                        if progress_callback:
                            progress_callback(
                                completed_count,
                                total_records,
                                f"已完成第 {completed_count}/{total_records} 条记录",
                                estimated_remaining_time
                            )

                except Exception as e:
                    print(f"❌ 处理任务结果时出错: {e}")

        # 保存结果
        final_results = [r for r in results if r is not None]
        result_df = pd.DataFrame(final_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_workflow_name = "".join(c for c in workflow_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_table_name = table_filename.split('.')[0]

        output_filename = f"{safe_workflow_name}_{safe_table_name}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_folder, output_filename)

        result_df.to_excel(output_path, index=False, engine='openpyxl')

        # 最终完成回调
        if progress_callback:
            progress_callback(total_records, total_records, "数据处理完成，正在保存文件", 0)

        print(f"✅ 处理完成，结果保存到: {output_path}")
        return output_path
