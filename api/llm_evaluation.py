from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import json
import yaml
import os
import shutil
from datetime import datetime
from pathlib import Path
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class LLMEvaluationAPI:
    def __init__(self):
        # 确保所有必要的目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            settings.CSV_TEST_PATH,
            settings.YAML_EVALUATE_PATH,
            settings.YAML_EVALUATE_TEMPLATES_PATH,
            settings.TEMP_RESULTS_PATH,
            settings.EVALUATION_RESULTS_PATH
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_workflows_config(self):
        """加载工作流配置"""
        config_path = settings.WORKFLOWS_CONFIG_PATH
        if not os.path.exists(config_path):
            # 创建默认配置
            default_config = {"workflows": []}
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载工作流配置失败: {e}")
            return {"workflows": []}

    def _save_workflows_config(self, config):
        """保存工作流配置"""
        config_path = settings.WORKFLOWS_CONFIG_PATH
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存工作流配置失败: {e}")
            raise HTTPException(status_code=500, detail="保存配置失败")


llm_eval_api = LLMEvaluationAPI()


# ==================== 配置被测大模型相关接口 ====================
@router.get("/models")
async def get_models():
    """获取所有已配置的大模型"""
    try:
        config = llm_eval_api._load_workflows_config()
        models = []

        for workflow in config.get("workflows", []):
            models.append({
                "id": workflow.get("id"),
                "name": workflow.get("name"),
                "url": workflow.get("url"),
                "api_key": workflow.get("api_key", ""),  # 确保有默认值
                "description": workflow.get("description", ""),
                "status": workflow.get("status", "active"),
                "created_at": workflow.get("created_at")
            })

        return JSONResponse({
            "status": "success",
            "data": models
        })
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取模型列表失败")


@router.post("/models")
async def create_model(
        name: str = Form(...),
        url: str = Form(...),
        api_key: str = Form(...),
        description: str = Form("")
):
    """创建新的大模型配置"""
    try:
        config = llm_eval_api._load_workflows_config()

        # 生成ID
        existing_ids = [w.get("id", "") for w in config.get("workflows", [])]
        model_id = f"model_{len(existing_ids) + 1}"
        while model_id in existing_ids:
            model_id = f"model_{len(existing_ids) + 1}_{datetime.now().strftime('%H%M%S')}"

        # 创建新模型配置
        new_model = {
            "id": model_id,
            "name": name,
            "url": url,
            "api_key": api_key,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        config.setdefault("workflows", []).append(new_model)
        llm_eval_api._save_workflows_config(config)

        return JSONResponse({
            "status": "success",
            "message": "模型配置创建成功",
            "data": new_model
        })
    except Exception as e:
        logger.error(f"创建模型配置失败: {e}")
        raise HTTPException(status_code=500, detail="创建模型配置失败")

@router.put("/models/{model_id}")
async def update_model(
        model_id: str,
        name: str = Form(...),
        url: str = Form(...),
        api_key: str = Form(...),
        description: str = Form("")
):
    """更新大模型配置"""
    try:
        config = llm_eval_api._load_workflows_config()
        workflows = config.get("workflows", [])

        found = False
        updated_model = None
        for i, model in enumerate(workflows):
            if model.get("id") == model_id:
                workflows[i]["name"] = name
                workflows[i]["url"] = url
                workflows[i]["api_key"] = api_key
                workflows[i]["description"] = description
                workflows[i]["updated_at"] = datetime.now().isoformat() # 记录更新时间
                updated_model = workflows[i]
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail="模型配置不存在")

        llm_eval_api._save_workflows_config(config)

        return JSONResponse({
            "status": "success",
            "message": "模型配置更新成功",
            "data": updated_model
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新模型配置失败: {e}")
        raise HTTPException(status_code=500, detail="更新模型配置失败")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """删除大模型配置"""
    try:
        config = llm_eval_api._load_workflows_config()
        workflows = config.get("workflows", [])

        # 查找并删除指定模型
        original_count = len(workflows)
        config["workflows"] = [w for w in workflows if w.get("id") != model_id]

        if len(config["workflows"]) == original_count:
            raise HTTPException(status_code=404, detail="模型配置不存在")

        llm_eval_api._save_workflows_config(config)

        return JSONResponse({
            "status": "success",
            "message": "模型配置删除成功"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型配置失败: {e}")
        raise HTTPException(status_code=500, detail="删除模型配置失败")


# ==================== 测评集管理相关接口 ====================

@router.get("/datasets")
async def get_datasets():
    """获取所有测评集"""
    try:
        datasets = []
        dataset_path = Path(settings.CSV_TEST_PATH)

        if dataset_path.exists():
            for file_path in dataset_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.xlsx']:
                    stat = file_path.stat()
                    datasets.append({
                        "name": file_path.name,
                        "size": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "file_type": file_path.suffix.lower()
                    })

        # 按修改时间排序
        datasets.sort(key=lambda x: x["modified_time"], reverse=True)

        return JSONResponse({
            "status": "success",
            "data": datasets
        })
    except Exception as e:
        logger.error(f"获取测评集列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取测评集列表失败")


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传测评集"""
    try:
        # 检查文件类型
        if not file.filename.lower().endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="只支持CSV和Excel文件")

        # 保存文件
        file_path = Path(settings.CSV_TEST_PATH) / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse({
            "status": "success",
            "message": "测评集上传成功",
            "data": {
                "filename": file.filename,
                "size": file_path.stat().st_size
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传测评集失败: {e}")
        raise HTTPException(status_code=500, detail="上传测评集失败")


@router.get("/datasets/{filename}/download")
async def download_dataset(filename: str):
    """下载测评集"""
    try:
        file_path = Path(settings.CSV_TEST_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载测评集失败: {e}")
        raise HTTPException(status_code=500, detail="下载测评集失败")


@router.delete("/datasets/{filename}")
async def delete_dataset(filename: str):
    """删除测评集"""
    try:
        file_path = Path(settings.CSV_TEST_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        file_path.unlink()

        return JSONResponse({
            "status": "success",
            "message": "测评集删除成功"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除测评集失败: {e}")
        raise HTTPException(status_code=500, detail="删除测评集失败")


# ==================== 测评规则管理相关接口 ====================

@router.get("/rules")
async def get_rules():
    """获取所有测评规则"""
    try:
        rules = []
        rules_path = Path(settings.YAML_EVALUATE_PATH)

        if rules_path.exists():
            for file_path in rules_path.glob("*.yaml"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        rule_content = yaml.safe_load(f)

                    stat = file_path.stat()
                    rule_info = {
                        "filename": file_path.name,
                        "name": rule_content.get("evaluation_rule", {}).get("name", file_path.stem),
                        "description": rule_content.get("evaluation_rule", {}).get("intro", ""),
                        "type": rule_content.get("evaluation_rule", {}).get("type", "unknown"),
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "size": stat.st_size
                    }
                    rules.append(rule_info)
                except Exception as e:
                    logger.warning(f"解析规则文件 {file_path.name} 失败: {e}")

        # 按修改时间排序
        rules.sort(key=lambda x: x["modified_time"], reverse=True)

        return JSONResponse({
            "status": "success",
            "data": rules
        })
    except Exception as e:
        logger.error(f"获取测评规则列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取测评规则列表失败")


@router.get("/rules/templates")
async def get_rule_templates():
    """获取测评规则模板"""
    try:
        templates = []
        templates_path = Path(settings.YAML_EVALUATE_TEMPLATES_PATH)

        if templates_path.exists():
            for file_path in templates_path.glob("*.yaml"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_content = yaml.safe_load(f)

                    template_info = {
                        "filename": file_path.name,
                        "name": template_content.get("evaluation_rule", {}).get("name", file_path.stem),
                        "type": template_content.get("evaluation_rule", {}).get("type", "prompt"),
                        "intro": template_content.get("evaluation_rule", {}).get("intro", ""),
                        "content": template_content
                    }
                    templates.append(template_info)
                except Exception as e:
                    logger.warning(f"解析模板文件 {file_path.name} 失败: {e}")

        return JSONResponse({
            "status": "success",
            "data": templates
        })
    except Exception as e:
        logger.error(f"获取测评规则模板失败: {e}")
        raise HTTPException(status_code=500, detail="获取测评规则模板失败")


@router.post("/rules")
async def create_rule(
        name: str = Form(...),
        rule_type: str = Form(...),
        intro: str = Form(...),
        model_name: str = Form("Doubao-pro-32k"),
        temperature: float = Form(0.3),
        prompt_description: str = Form(...)
):
    """创建测评规则"""
    try:
        # 构建YAML内容
        rule_content = {
            "model_config": {
                "name": model_name,
                "temperature": temperature
            },
            "evaluation_rule": {
                "name": name,
                "type": rule_type,
                "intro": intro,
                "prompt": {
                    "role": {
                        "description": prompt_description
                    }
                }
            }
        }

        # 生成文件名
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_name}.yaml"
        file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        # 如果文件已存在，添加时间戳
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}.yaml"
            file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(rule_content, f, allow_unicode=True, default_flow_style=False, indent=2)

        return JSONResponse({
            "status": "success",
            "message": "测评规则创建成功",
            "data": {
                "filename": filename,
                "name": name
            }
        })
    except Exception as e:
        logger.error(f"创建测评规则失败: {e}")
        raise HTTPException(status_code=500, detail="创建测评规则失败")


@router.get("/rules/{filename}")
async def get_rule_detail(filename: str):
    """获取测评规则详情"""
    try:
        file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="规则文件不存在")

        with open(file_path, 'r', encoding='utf-8') as f:
            rule_content = yaml.safe_load(f)

        return JSONResponse({
            "status": "success",
            "data": rule_content
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取测评规则详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取测评规则详情失败")


@router.put("/rules/{filename}")
async def update_rule(
        filename: str,
        name: str = Form(...),
        rule_type: str = Form(...),
        intro: str = Form(...),
        model_name: str = Form("Doubao-pro-32k"),
        temperature: float = Form(0.3),
        prompt_description: str = Form(...)
):
    """更新测评规则"""
    try:
        file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="规则文件不存在")

        # 构建更新后的YAML内容
        rule_content = {
            "model_config": {
                "name": model_name,
                "temperature": temperature
            },
            "evaluation_rule": {
                "name": name,
                "type": rule_type,
                "intro": intro,
                "prompt": {
                    "role": {
                        "description": prompt_description
                    }
                }
            }
        }

        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(rule_content, f, allow_unicode=True, default_flow_style=False, indent=2)

        return JSONResponse({
            "status": "success",
            "message": "测评规则更新成功"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新测评规则失败: {e}")
        raise HTTPException(status_code=500, detail="更新测评规则失败")


@router.get("/rules/{filename}/download")
async def download_rule(filename: str):
    """下载测评规则"""
    try:
        file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载测评规则失败: {e}")
        raise HTTPException(status_code=500, detail="下载测评规则失败")


@router.delete("/rules/{filename}")
async def delete_rule(filename: str):
    """删除测评规则"""
    try:
        file_path = Path(settings.YAML_EVALUATE_PATH) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        file_path.unlink()

        return JSONResponse({
            "status": "success",
            "message": "测评规则删除成功"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除测评规则失败: {e}")
        raise HTTPException(status_code=500, detail="删除测评规则失败")


# 在api/llm_evaluation.py文件末尾添加以下代码

import asyncio
import threading
import uuid
from typing import Dict, Any
from services.testqa_service import DifyWorkflowCaller,ModelEvaluator

# 全局任务状态存储
running_tasks: Dict[str, Dict[str, Any]] = {}


class TaskProgress:
    def __init__(self, task_id: str, task_type: str, total_steps: int):
        self.task_id = task_id
        self.task_type = task_type  # "test" 或 "evaluation"
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "running"  # running, completed, failed
        self.message = "任务开始"
        self.result_file = None
        self.error_message = None

        # 详细进度信息
        self.current_record = 0
        self.total_records = 0
        self.detailed_message = ""
        self.estimated_time = 0  # 新增：预估剩余时间（秒）

    def update(self, step: int = None, message: str = None, status: str = None,
               current_record: int = None, total_records: int = None,
               detailed_message: str = None, estimated_time: float = None):
        if step is not None:
            self.current_step = step
        if message is not None:
            self.message = message
        if status is not None:
            self.status = status
        if current_record is not None:
            self.current_record = current_record
        if total_records is not None:
            self.total_records = total_records
        if detailed_message is not None:
            self.detailed_message = detailed_message
        if estimated_time is not None:
            self.estimated_time = estimated_time

    def to_dict(self):
        # 计算整体进度
        base_progress = 0
        if self.total_steps > 0:
            base_progress = int((self.current_step / self.total_steps) * 100)

        # 如果有记录级别的进度，进行细化计算
        detailed_progress = base_progress
        if self.total_records > 0 and self.current_step > 0:
            # 每个大步骤内的细分进度
            step_weight = 100 / self.total_steps
            current_step_base = (self.current_step - 1) * step_weight
            record_progress = (self.current_record / self.total_records) * step_weight
            detailed_progress = min(100, int(current_step_base + record_progress))

        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": detailed_progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message": self.message,
            "detailed_message": self.detailed_message,
            "current_record": self.current_record,
            "total_records": self.total_records,
            "result_file": self.result_file,
            "error_message": self.error_message,
            "estimated_time": self.estimated_time  # 新增
        }


def run_test_task(task_id: str, model_name: str, dataset_filename: str, max_workers: int = 5):
    """运行测评任务（后台线程）"""
    progress = running_tasks[task_id]

    def progress_callback(current_record, total_records, message, estimated_time):
        """进度回调函数"""
        progress.update(
            step=2,  # 处理数据集步骤
            current_record=current_record,
            total_records=total_records,
            detailed_message=message,
            estimated_time=estimated_time
        )

    try:
        # 初始化调用器
        progress.update(1, "初始化工作流调用器...")
        caller = DifyWorkflowCaller()

        progress.update(2, f"开始处理数据集: {dataset_filename}")

        # 执行测评（传递进度回调和并发数）
        result_file = caller.process_table(dataset_filename, model_name, progress_callback, max_workers)

        progress.update(3, "测评任务完成", "completed")
        progress.result_file = os.path.basename(result_file)

    except Exception as e:
        logger.error(f"测评任务失败: {e}")
        progress.update(status="failed", message="测评任务失败")
        progress.error_message = str(e)


def run_evaluation_task(task_id: str, rule_filename: str, result_filename: str, max_workers: int = 3):
    """运行评价任务（后台线程）"""
    progress = running_tasks[task_id]

    def progress_callback(current_record, total_records, message, estimated_time):
        """进度回调函数"""
        progress.update(
            step=3,  # 评价记录步骤
            current_record=current_record,
            total_records=total_records,
            detailed_message=message,
            estimated_time=estimated_time
        )

    try:
        # 初始化评价器
        progress.update(1, "初始化模型评价器...")
        evaluator = ModelEvaluator()

        progress.update(2, f"加载测评规则: {rule_filename}")

        # 构建结果文件的完整路径
        result_file_path = os.path.join(settings.TEMP_RESULTS_PATH, result_filename)

        progress.update(3, f"开始评价: {result_filename}")

        # 执行评价（传递进度回调和并发数）
        evaluation_file = evaluator.evaluate_results(result_file_path, rule_filename, progress_callback, max_workers)

        progress.update(4, "评价任务完成", "completed")
        progress.result_file = os.path.basename(evaluation_file)

    except Exception as e:
        logger.error(f"评价任务失败: {e}")
        progress.update(status="failed", message="评价任务失败")
        progress.error_message = str(e)


# ==================== 测评任务相关接口 ====================

@router.get("/evaluation/results")
async def get_evaluation_results():
    """获取所有测评结果文件"""
    try:
        results = []

        # 获取测评结果（测）
        temp_results_path = Path(settings.TEMP_RESULTS_PATH)
        if temp_results_path.exists():
            for file_path in temp_results_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.csv']:
                    stat = file_path.stat()
                    results.append({
                        "filename": file_path.name,
                        "type": "测",
                        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "size": stat.st_size,
                        "folder": "temp_results"
                    })

        # 获取评价结果（评）
        evaluation_results_path = Path(settings.EVALUATION_RESULTS_PATH)
        if evaluation_results_path.exists():
            for file_path in evaluation_results_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.csv']:
                    stat = file_path.stat()
                    results.append({
                        "filename": file_path.name,
                        "type": "评",
                        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "size": stat.st_size,
                        "folder": "evaluation_results"
                    })

        # 按创建时间排序
        results.sort(key=lambda x: x["created_time"], reverse=True)

        return JSONResponse({
            "status": "success",
            "data": results
        })
    except Exception as e:
        logger.error(f"获取结果文件列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取结果文件列表失败")


@router.post("/evaluation/tasks")
async def create_evaluation_task(
        task_type: str = Form(...),  # "test" 或 "evaluation"
        model_name: Optional[str] = Form(None),  # 测评任务需要
        dataset_filename: Optional[str] = Form(None),  # 都需要
        rule_filename: Optional[str] = Form(None),  # 评价任务需要
        result_filename: Optional[str] = Form(None),  # 评价任务需要
        max_workers: int = Form(5)  # 新增：并发数
):
    """创建测评任务"""
    try:
        # 验证并发数
        if max_workers < 1 or max_workers > 20:
            raise HTTPException(status_code=400, detail="并发数必须在1-20之间")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        if task_type == "test":
            if not model_name or not dataset_filename:
                raise HTTPException(status_code=400, detail="测评任务需要模型名称和数据集文件名")

            # ... 现有验证代码 ...

            # 创建进度跟踪
            progress = TaskProgress(task_id, "test", 3)
            running_tasks[task_id] = progress

            # 在后台运行任务
            thread = threading.Thread(
                target=run_test_task,
                args=(task_id, model_name, dataset_filename, max_workers)
            )
            thread.daemon = True
            thread.start()

        elif task_type == "evaluation":
            if not rule_filename or not result_filename:
                raise HTTPException(status_code=400, detail="评价任务需要规则文件名和结果文件名")

            # 验证规则文件是否存在
            rule_path = Path(settings.YAML_EVALUATE_PATH) / rule_filename
            if not rule_path.exists():
                raise HTTPException(status_code=400, detail="指定的规则文件不存在")

            # 验证结果文件是否存在
            result_path = Path(settings.TEMP_RESULTS_PATH) / result_filename
            if not result_path.exists():
                raise HTTPException(status_code=400, detail="指定的结果文件不存在")

            # 创建进度跟踪
            progress = TaskProgress(task_id, "evaluation", 4)
            running_tasks[task_id] = progress

            # 在后台运行任务
            thread = threading.Thread(
                target=run_evaluation_task,
                args=(task_id, rule_filename, result_filename, max_workers)
            )
            thread.daemon = True
            thread.start()

        else:
            raise HTTPException(status_code=400, detail="无效的任务类型")

        return JSONResponse({
            "status": "success",
            "message": "任务创建成功",
            "data": {
                "task_id": task_id,
                "task_type": task_type,
                "max_workers": max_workers
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建测评任务失败: {e}")
        raise HTTPException(status_code=500, detail="创建测评任务失败")


@router.get("/evaluation/tasks/{task_id}/progress")
async def get_task_progress(task_id: str):
    """获取任务进度"""
    try:
        if task_id not in running_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")

        progress = running_tasks[task_id]
        return JSONResponse({
            "status": "success",
            "data": progress.to_dict()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务进度失败: {e}")
        raise HTTPException(status_code=500, detail="获取任务进度失败")


@router.get("/evaluation/results/{filename}/download")
async def download_result_file(filename: str):
    """下载结果文件"""
    try:
        # 先在temp_results中查找
        temp_path = Path(settings.TEMP_RESULTS_PATH) / filename
        if temp_path.exists():
            return FileResponse(
                path=str(temp_path),
                filename=filename,
                media_type='application/octet-stream'
            )

        # 再在evaluation_results中查找
        eval_path = Path(settings.EVALUATION_RESULTS_PATH) / filename
        if eval_path.exists():
            return FileResponse(
                path=str(eval_path),
                filename=filename,
                media_type='application/octet-stream'
            )

        raise HTTPException(status_code=404, detail="文件不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail="下载文件失败")


@router.delete("/evaluation/results/{filename}")
async def delete_result_file(filename: str):
    """删除结果文件"""
    try:
        deleted = False

        # 先在temp_results中查找
        temp_path = Path(settings.TEMP_RESULTS_PATH) / filename
        if temp_path.exists():
            temp_path.unlink()
            deleted = True

        # 再在evaluation_results中查找
        eval_path = Path(settings.EVALUATION_RESULTS_PATH) / filename
        if eval_path.exists():
            eval_path.unlink()
            deleted = True

        if not deleted:
            raise HTTPException(status_code=404, detail="文件不存在")

        return JSONResponse({
            "status": "success",
            "message": "文件删除成功"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        raise HTTPException(status_code=500, detail="删除文件失败")


@router.get("/evaluation/temp-results")
async def get_temp_results():
    """获取测评结果文件（用于评价任务选择）"""
    try:
        results = []
        temp_results_path = Path(settings.TEMP_RESULTS_PATH)

        if temp_results_path.exists():
            for file_path in temp_results_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.csv']:
                    stat = file_path.stat()
                    results.append({
                        "filename": file_path.name,
                        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "size": stat.st_size
                    })

        # 按创建时间排序
        results.sort(key=lambda x: x["created_time"], reverse=True)

        return JSONResponse({
            "status": "success",
            "data": results
        })
    except Exception as e:
        logger.error(f"获取测评结果文件失败: {e}")
        raise HTTPException(status_code=500, detail="获取测评结果文件失败")


