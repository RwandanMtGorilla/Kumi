from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# 比较操作符枚举
class ComparisonOperator(str, Enum):
    CONTAINS = "contains"
    NOT_CONTAINS = "not contains"
    START_WITH = "start with"
    END_WITH = "end with"
    IS = "is"
    IS_NOT = "is not"
    EMPTY = "empty"
    NOT_EMPTY = "not empty"
    EQUAL = "="
    NOT_EQUAL = "≠"
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = "≥"
    LESS_EQUAL = "≤"
    BEFORE = "before"
    AFTER = "after"

# 逻辑操作符枚举
class LogicalOperator(str, Enum):
    AND = "and"
    OR = "or"

# 筛选条件
class MetadataCondition(BaseModel):
    name: List[str] = Field(..., description="需要筛选的metadata名称")
    comparison_operator: ComparisonOperator = Field(..., description="比较操作符")
    value: Optional[str] = Field(None, description="对比值")

# 元数据筛选
class MetadataFilter(BaseModel):
    logical_operator: LogicalOperator = Field(LogicalOperator.AND, description="逻辑操作符")
    conditions: List[MetadataCondition] = Field(..., description="条件列表")

# 检索设置
class RetrievalSetting(BaseModel):
    top_k: int = Field(..., ge=1, le=100, description="检索结果的最大数量")
    score_threshold: float = Field(..., ge=0, le=1, description="结果与查询相关性的分数限制")

# 检索请求
class RetrievalRequest(BaseModel):
    knowledge_id: str = Field(..., description="知识库唯一ID")
    query: str = Field(..., description="用户的查询")
    retrieval_setting: RetrievalSetting = Field(..., description="知识检索参数")
    metadata_condition: Optional[MetadataFilter] = Field(None, description="元数据筛选")

# 检索记录
class RetrievalRecord(BaseModel):
    content: str = Field(..., description="包含知识库中数据源的文本块")
    score: float = Field(..., ge=0, le=1, description="结果与查询的相关性分数")
    title: str = Field(..., description="文档标题")
    metadata: Optional[Dict[str, Any]] = Field(None, description="包含数据源中文档的元数据属性及其值")

# 检索响应
class RetrievalResponse(BaseModel):
    records: List[RetrievalRecord] = Field(..., description="从知识库查询的记录列表")

# 错误响应
class ErrorResponse(BaseModel):
    error_code: int = Field(..., description="错误代码")
    error_msg: str = Field(..., description="API异常描述")
