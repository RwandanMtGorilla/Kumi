// 知识库测试页面 JavaScript

// ============================================================================
// 全局变量定义区
// ============================================================================

// --- 原始数据相关 ---
let originalMatrix = null;
let filteredMatrix = null;
let originalXData = [];
let originalYData = [];
let currentXData = [];
let currentYData = [];
let currentXLabels = [];
let currentYLabels = [];
let currentResult = null;
let xAvailableFields = [];
let yAvailableFields = [];
let availableCollections = [];

// --- 多图管理相关 ---
let allSimilarityResults = []; // 存储所有相似度矩阵结果，每个元素包含 visualConfig
let currentMatrixIndex = 0; // 当前显示的矩阵索引（已废弃，保留兼容）
let selectedMatrixIndices = []; // 存储选中的矩阵索引（已废弃，保留兼容）
let differenceMatrices = {}; // 存储差值矩阵的缓存 key格式: "idx1-idx2"

// --- 全局默认配色（非可视化控制值的一部分，是全局UI设置） ---
let currentColorScheme = 'viridis';

// ============================================================================
// 新架构：全局UI状态管理（与图表配置完全分离）
// ============================================================================

// 全局UI状态（独立于任何图表）
let globalUIState = {
    // 数值来源：应用数据按钮控制
    dataSource: {
        primaryIndex: null,      // 主数据源索引（第一个按下应用数据的图）
        subtractIndex: null,     // 减数索引（第二个按下应用数据的图，差值模式）
        currentMatrix: null,     // 当前显示的矩阵数据（原始或差值）
        currentXData: [],
        currentYData: [],
        currentXLabels: [],
        currentYLabels: [],
        xAvailableFields: [],
        yAvailableFields: []
    },

    // 字段配置：跟随主数据源，可独立修改
    displayFields: {
        xField: null,
        yField: null
    },

    // 筛选器：可以来自多个图（或逻辑）
    filters: {
        activeFilterIndices: [],  // 应用筛选器按钮选中的图表索引
        // 当前UI显示的筛选器值（独占模式下可编辑）
        uiState: {
            similarityRange: { min: 0, max: 1 },
            topK: { value: 0, axis: 'x' }
        }
    },

    // 排序：跟随字段配置，可独立修改
    sorting: {
        order: 'none'
    },

    // 独占模式
    exclusiveMode: {
        active: false,
        editingIndex: null  // 正在编辑的图表索引
    }
};

// 每张图的按钮状态
let matrixButtonStates = [];
// 结构示例：
// {
//     index: 0,
//     applyData: false,      // 应用数据按钮
//     applyFilter: false,    // 应用筛选器按钮
//     exclusive: false       // 独占模式按钮
// }


const DEFAULT_FIELD_NAMES = ['document', 'text', 'name'];

// API配置 - 直接使用当前系统的API端点
const API_BASE_URL = '/api/knowledge/similarity';

// ============================================================================
// 可视化配置管理
// ============================================================================

/**
 * 创建默认的可视化配置对象
 * 每张图都有独立的可视化控制值
 */
function createDefaultVisualConfig(xAvailableFields, yAvailableFields) {
    const defaultXField = getDefaultDisplayField(xAvailableFields);
    const defaultYField = getDefaultDisplayField(yAvailableFields);

    return {
        // 1. 显示字段配置
        displayFields: {
            xField: defaultXField,  // 横坐标显示字段
            yField: defaultYField   // 纵坐标显示字段
        },

        // 2. 显示值范围配置（相似度阈值）
        similarityRange: {
            min: 0,
            max: 1
        },

        // 3. 筛选器配置
        filters: {
            topK: {
                value: 0,           // Top-K值，0表示显示全部
                axis: 'x'           // 'x' 或 'y'
            }
            // 注意：阈值范围本身也起到筛选作用，已在 similarityRange 中定义
        },

        // 4. 排序配置
        sorting: {
            order: 'none'  // 'none', 'asc', 'desc', 'x_asc', 'x_desc', 'y_asc', 'y_desc'
        },

        // 5. 布尔矩阵缓存（新增）
        cachedMasks: {
            thresholdMask: null,    // 阈值筛选的布尔矩阵
            topKMask: null,         // Top-K筛选的布尔矩阵
            finalMask: null         // AND合并后的最终遮罩
        }
    };
}

/**
 * 从当前UI控件获取可视化配置
 */
function getCurrentVisualConfigFromUI() {
    return {
        displayFields: {
            xField: document.getElementById('xDisplayField').value,
            yField: document.getElementById('yDisplayField').value
        },
        similarityRange: {
            min: parseFloat(document.getElementById('minSimilaritySlider').value),
            max: parseFloat(document.getElementById('maxSimilaritySlider').value)
        },
        filters: {
            topK: {
                value: parseInt(document.getElementById('topkSlider').value),
                axis: currentTopkAxis
            }
        },
        sorting: {
            order: document.getElementById('sortOrder').value
        }
    };
}

/**
 * 将可视化配置应用到UI控件
 */
function applyVisualConfigToUI(visualConfig) {
    console.log('[应用配置到UI]', {
        displayFields: visualConfig.displayFields,
        similarityRange: visualConfig.similarityRange,
        topK: visualConfig.filters.topK,
        sorting: visualConfig.sorting
    });

    // 1. 更新显示字段
    document.getElementById('xDisplayField').value = visualConfig.displayFields.xField;
    document.getElementById('yDisplayField').value = visualConfig.displayFields.yField;

    // 2. 更新相似度范围滑块
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    minSlider.value = visualConfig.similarityRange.min;
    maxSlider.value = visualConfig.similarityRange.max;
    minInput.value = visualConfig.similarityRange.min;
    maxInput.value = visualConfig.similarityRange.max;

    console.log('[应用配置] 设置滑块值:', minSlider.value, '-', maxSlider.value);

    document.getElementById('minSimilarityValue').textContent = visualConfig.similarityRange.min.toFixed(2);
    document.getElementById('maxSimilarityValue').textContent = visualConfig.similarityRange.max.toFixed(2);

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    const percent1 = (visualConfig.similarityRange.min - parseFloat(minSlider.min)) /
                     (parseFloat(minSlider.max) - parseFloat(minSlider.min)) * 100;
    const percent2 = (visualConfig.similarityRange.max - parseFloat(maxSlider.min)) /
                     (parseFloat(maxSlider.max) - parseFloat(maxSlider.min)) * 100;
    track.style.left = percent1 + '%';
    track.style.width = (percent2 - percent1) + '%';

    // 3. 更新Top-K筛选器
    document.getElementById('topkSlider').value = visualConfig.filters.topK.value;
    currentTopkAxis = visualConfig.filters.topK.axis;

    // 更新Top-K显示
    updateTopkDisplay();

    // 更新轴选择按钮状态
    document.getElementById('xAxisBtn').classList.toggle('active', currentTopkAxis === 'x');
    document.getElementById('yAxisBtn').classList.toggle('active', currentTopkAxis === 'y');

    // 4. 更新排序
    document.getElementById('sortOrder').value = visualConfig.sorting.order;
}

// ============================================================================
// 常量配置
// ============================================================================

// 颜色方案配置
const colorSchemes = {
    viridis: 'Viridis',
    plasma: 'Plasma',
    cividis: 'Cividis',
    hot: 'Hot',
    YlGnBu: 'YlGnBu',

};

// ============================================================================
// UI辅助函数
// ============================================================================

// 消息显示函数
function showMessage(type, message, duration = 5000) {
    const messageId = type + 'Message';
    const messageDiv = document.getElementById(messageId);
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';

    // 隐藏其他类型的消息
    ['error', 'success', 'warning', 'info'].forEach(t => {
        if (t !== type) {
            document.getElementById(t + 'Message').style.display = 'none';
        }
    });

    if (duration > 0) {
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, duration);
    }
}

function showError(message, duration = 5000) { showMessage('error', message, duration); }
function showSuccess(message, duration = 3000) { showMessage('success', message, duration); }
function showWarning(message, duration = 4000) { showMessage('warning', message, duration); }
function showInfo(message, duration = 3000) { showMessage('info', message, duration); }

// 显示加载状态
function showLoading(show = true, text = '正在加载...') {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
    document.getElementById('loadingText').textContent = text;
}

// 显示必要字段验证错误
function showFieldValidationError(fieldId, show = true) {
    const field = document.getElementById(fieldId);
    const errorDiv = document.getElementById(fieldId + 'Error');

    if (show) {
        field.classList.add('required-empty');
        errorDiv.style.display = 'block';
    } else {
        field.classList.remove('required-empty');
        errorDiv.style.display = 'none';
    }
}

// 清除所有验证错误状态
function clearValidationErrors() {
    document.getElementById('xCollectionError').style.display = 'none';
    document.getElementById('yCollectionError').style.display = 'none';
}

// 获取默认显示字段函数
function getDefaultDisplayField(availableFields) {
    // 遍历优先级列表，返回第一个匹配的字段
    for (const defaultField of DEFAULT_FIELD_NAMES) {
        if (availableFields.includes(defaultField)) {
            return defaultField;
        }
    }
    // 如果没有匹配的默认字段，返回 'order_id'
    return 'order_id';
}

// ============================================================================
// API调用与数据加载
// ============================================================================

// API调用函数
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' }
    };

    const finalOptions = { ...defaultOptions, ...options };

    try {
        const response = await fetch(url, finalOptions);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.detail || `HTTP ${response.status}`);
        }

        if (data.success === false) {
            throw new Error(data.error || '未知错误');
        }

        return data;
    } catch (error) {
        console.error('API调用失败:', error);
        throw error;
    }
}

// 修改loadCollections函数
async function loadCollections() {
    try {
        showLoading(true, '正在加载Collections...');

        const data = await apiCall('/collections');
        const collections = data.collections || [];

        // 更新全局变量
        availableCollections = collections;

        // 更新所有现有的选择框
        updateAllCollectionSelects();

        showSuccess(`成功加载 ${collections.length} 个Collections`);

    } catch (error) {
        showError('加载Collections失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 新增：更新所有collection选择框
function updateAllCollectionSelects() {
    // 更新所有X轴选择框
    const xSelects = document.querySelectorAll('.x-collection-select');
    xSelects.forEach(select => {
        updateSingleCollectionSelect(select);
    });

    // 更新所有Y轴选择框
    const ySelects = document.querySelectorAll('.y-collection-select');
    ySelects.forEach(select => {
        updateSingleCollectionSelect(select);
    });
}

// 新增：更新单个collection选择框
function updateSingleCollectionSelect(select) {
    const currentValue = select.value;
    select.innerHTML = '<option value="">请选择...</option>';

    availableCollections.forEach(collection => {
        const option = new Option(collection, collection);
        select.add(option);
    });

    // 如果之前有选中值且该值仍然存在，保持选中
    if (currentValue && availableCollections.includes(currentValue)) {
        select.value = currentValue;
    }
}

// 修改addCollectionSelect函数
function addCollectionSelect(axis) {
    const containerId = axis === 'x' ? 'xCollectionContainer' : 'yCollectionContainer';
    const container = document.getElementById(containerId);
    const selectClass = axis === 'x' ? 'x-collection-select' : 'y-collection-select';

    // 创建新的选择框行
    const newRow = document.createElement('div');
    newRow.className = 'collection-select-row';

    // 创建select元素
    const newSelect = document.createElement('select');
    newSelect.className = selectClass;

    // 使用全局变量填充options
    updateSingleCollectionSelect(newSelect);

    // 创建删除按钮
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn-remove-collection';
    removeBtn.textContent = '-';
    removeBtn.title = '删除';
    removeBtn.onclick = function() {
        removeCollectionSelect(this);
    };

    newRow.appendChild(newSelect);
    newRow.appendChild(removeBtn);
    container.appendChild(newRow);

    // 清除验证错误
    const errorId = axis === 'x' ? 'xCollectionError' : 'yCollectionError';
    showFieldValidationError(errorId, false);
}

// 删除collection选择框
function removeCollectionSelect(button) {
    const row = button.closest('.collection-select-row');
    row.remove();
}


// ============================================================================
// 布尔矩阵计算函数（新架构核心）
// ============================================================================

/**
 * 计算阈值筛选的布尔矩阵
 * @param {Array<Array<number>>} matrix - 原始相似度矩阵
 * @param {number} minSim - 最小相似度阈值
 * @param {number} maxSim - 最大相似度阈值
 * @returns {Array<Array<boolean>>} - 布尔矩阵，true表示通过筛选
 */
function computeThresholdMask(matrix, minSim, maxSim) {
    console.log(`[布尔矩阵] 计算阈值遮罩: ${minSim.toFixed(2)} - ${maxSim.toFixed(2)}`);
    return matrix.map(row =>
        row.map(val => {
            if (val === null || val === undefined) return false;
            return val >= minSim && val <= maxSim;
        })
    );
}

/**
 * 计算Top-K筛选的布尔矩阵
 * @param {Array<Array<number>>} matrix - 原始相似度矩阵
 * @param {number} topK - Top-K值，0表示全部显示
 * @param {string} axis - 'x' 或 'y'
 * @returns {Array<Array<boolean>>} - 布尔矩阵，true表示是Top-K
 */
function computeTopKMask(matrix, topK, axis) {
    console.log(`[布尔矩阵] 计算Top-K遮罩: Top-${topK}, 轴: ${axis}`);

    if (topK === 0) {
        // Top-K为0，全部通过
        return matrix.map(row => row.map(() => true));
    }

    // 初始化布尔矩阵为false
    const mask = matrix.map(row => row.map(() => false));

    if (axis === 'x') {
        // 横轴Top-K：对每一行（Y轴的每个项目），找出相似度最高的K个X轴项目
        for (let i = 0; i < matrix.length; i++) {
            const row = matrix[i];
            const validPairs = [];

            for (let j = 0; j < row.length; j++) {
                const val = row[j];
                if (val !== null && val !== undefined) {
                    validPairs.push({ index: j, similarity: val });
                }
            }

            // 按相似度降序排序，取前K个
            validPairs.sort((a, b) => b.similarity - a.similarity);
            const topKPairs = validPairs.slice(0, topK);

            // 在布尔矩阵中标记Top-K位置为true
            topKPairs.forEach(pair => {
                mask[i][pair.index] = true;
            });
        }
    } else {
        // 纵轴Top-K：对每一列（X轴的每个项目），找出相似度最高的K个Y轴项目
        for (let j = 0; j < matrix[0].length; j++) {
            const validPairs = [];

            for (let i = 0; i < matrix.length; i++) {
                const val = matrix[i][j];
                if (val !== null && val !== undefined) {
                    validPairs.push({ index: i, similarity: val });
                }
            }

            // 按相似度降序排序，取前K个
            validPairs.sort((a, b) => b.similarity - a.similarity);
            const topKPairs = validPairs.slice(0, topK);

            // 在布尔矩阵中标记Top-K位置为true
            topKPairs.forEach(pair => {
                mask[pair.index][j] = true;
            });
        }
    }

    return mask;
}

/**
 * 单图内AND合并：阈值遮罩 AND Top-K遮罩
 * @param {Array<Array<boolean>>} mask1 - 第一个布尔矩阵
 * @param {Array<Array<boolean>>} mask2 - 第二个布尔矩阵
 * @returns {Array<Array<boolean>>} - AND运算后的布尔矩阵
 */
function combineWithAND(mask1, mask2) {
    console.log('[布尔矩阵] 执行AND合并');
    return mask1.map((row, i) =>
        row.map((val, j) => val && mask2[i][j])
    );
}

/**
 * 多图间OR合并：多个最终遮罩的OR运算
 * @param {Array<Array<Array<boolean>>>} masks - 多个布尔矩阵
 * @returns {Array<Array<boolean>>} - OR运算后的布尔矩阵
 */
function combineWithOR(masks) {
    if (masks.length === 0) {
        console.warn('[布尔矩阵] OR合并：没有输入遮罩');
        return null;
    }

    if (masks.length === 1) {
        console.log('[布尔矩阵] OR合并：只有一个遮罩，直接返回');
        return masks[0];
    }

    console.log(`[布尔矩阵] 执行OR合并：${masks.length}个遮罩`);

    return masks.reduce((result, mask) =>
        result.map((row, i) =>
            row.map((val, j) => val || mask[i][j])
        )
    );
}

/**
 * 应用布尔遮罩到原始矩阵
 * @param {Array<Array<number>>} originalMatrix - 原始数据矩阵
 * @param {Array<Array<boolean>>} mask - 布尔遮罩
 * @returns {Array<Array<number|null>>} - 应用遮罩后的矩阵
 */
function applyMask(originalMatrix, mask) {
    console.log('[布尔矩阵] 应用遮罩到原始数据');
    return originalMatrix.map((row, i) =>
        row.map((val, j) => mask[i][j] ? val : null)
    );
}

/**
 * 计算单张图的最终遮罩（阈值 AND Top-K）
 * @param {number} index - 图表索引
 * @returns {Array<Array<boolean>>|null} - 最终布尔遮罩
 */
function computeFinalMaskForMatrix(index) {
    if (index < 0 || index >= allSimilarityResults.length) {
        console.error(`[布尔矩阵] 无效的图表索引: ${index}`);
        return null;
    }

    const matrixData = allSimilarityResults[index];
    const config = matrixData.visualConfig;
    const originalMatrix = matrixData.matrix;

    console.log(`[布尔矩阵] 计算图 ${index} 的最终遮罩`);

    // 1. 计算阈值遮罩
    const thresholdMask = computeThresholdMask(
        originalMatrix,
        config.similarityRange.min,
        config.similarityRange.max
    );

    // 2. 计算Top-K遮罩
    const topKMask = computeTopKMask(
        originalMatrix,
        config.filters.topK.value,
        config.filters.topK.axis
    );

    // 3. AND合并
    const finalMask = combineWithAND(thresholdMask, topKMask);

    // 4. 缓存到配置中
    if (!config.cachedMasks) {
        config.cachedMasks = {};
    }
    config.cachedMasks.thresholdMask = thresholdMask;
    config.cachedMasks.topKMask = topKMask;
    config.cachedMasks.finalMask = finalMask;

    return finalMask;
}

// ============================================================================
// 数据处理与标签生成
// ============================================================================

// 生成唯一标签函数，使用零宽字符和不间断空格确保唯一性
function generateUniqueLabels(data, field) {
    // 使用 Map 来跟踪每个值的出现次数
    const valueCountMap = new Map();

    return data.map((item, index) => {
        let baseValue;
        if (field === 'order_id') {
            baseValue = `ID-${item[field]}`;
        } else {
            baseValue = String(item[field] || 'N/A');
        }

        // 跟踪值的出现次数
        const currentCount = valueCountMap.get(baseValue) || 0;
        valueCountMap.set(baseValue, currentCount + 1);

        // 如果值重复，添加零宽字符和不间断空格来确保唯一性
        if (currentCount > 0) {
            // 零宽字符：\u200B (零宽空格)
            // 不间断空格：\u00A0
            // 根据重复次数添加不同数量的零宽字符
            const uniqueSuffix = '\u200B'.repeat(currentCount) + '\u00A0';
            return baseValue + uniqueSuffix;
        } else {
            return baseValue;
        }
    });
}

// 更新显示字段选择器
function updateDisplayFieldControls() {
    if (!currentResult) return;

    // 显示字段控制区域
    document.getElementById('displayFieldControls').style.display = 'block';
    document.getElementById('yDisplayFieldControls').style.display = 'block';

    // 获取默认显示字段
    const defaultXField = getDefaultDisplayField(xAvailableFields);
    const defaultYField = getDefaultDisplayField(yAvailableFields);

    // 更新X轴显示字段列表
    const xDisplaySelect = document.getElementById('xDisplayField');
    xDisplaySelect.innerHTML = '';
    xAvailableFields.forEach(field => {
        const option = new Option(
            field === 'order_id' ? '顺序ID' : field,
            field
        );
        xDisplaySelect.add(option);
    });
    xDisplaySelect.value = defaultXField; // 使用匹配到的默认字段

    // 更新Y轴显示字段列表
    const yDisplaySelect = document.getElementById('yDisplayField');
    yDisplaySelect.innerHTML = '';
    yAvailableFields.forEach(field => {
        const option = new Option(
            field === 'order_id' ? '顺序ID' : field,
            field
        );
        yDisplaySelect.add(option);
    });
    yDisplaySelect.value = defaultYField; // 使用匹配到的默认字段

    // 添加字段选择变化监听器
    xDisplaySelect.addEventListener('change', function() {
        updateLabelsAndVisualization();
    });

    yDisplaySelect.addEventListener('change', function() {
        updateLabelsAndVisualization();
    });
}


// 更新标签和可视化（新架构）
function updateLabelsAndVisualization() {
    // 使用全局状态
    if (!globalUIState.dataSource.currentMatrix) return;

    const xField = document.getElementById('xDisplayField').value;
    const yField = document.getElementById('yDisplayField').value;

    // 更新全局状态的显示字段
    globalUIState.displayFields.xField = xField;
    globalUIState.displayFields.yField = yField;

    // *** 只在独占模式下才保存到图表配置 ***
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
        config.displayFields.xField = xField;
        config.displayFields.yField = yField;
        console.log(`[独占模式] 保存显示字段: xField=${xField}, yField=${yField}`);
    }

    // 使用全局状态的数据生成唯一标签
    globalUIState.dataSource.currentXLabels = generateUniqueLabels(
        globalUIState.dataSource.currentXData,
        xField
    );
    globalUIState.dataSource.currentYLabels = generateUniqueLabels(
        globalUIState.dataSource.currentYData,
        yField
    );

    // 同步到旧变量（兼容性）
    currentXLabels = globalUIState.dataSource.currentXLabels;
    currentYLabels = globalUIState.dataSource.currentYLabels;

    // 更新热力图
    updateHeatmap();
}

// ============================================================================
// 可视化控制与筛选
// ============================================================================

// 设置Top-K轴选择
function setTopkAxis(axis) {
    currentTopkAxis = axis;

    // 更新全局筛选器状态
    globalUIState.filters.uiState.topK.axis = axis;

    // *** 只在独占模式下才保存到图表配置 ***
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
        config.filters.topK.axis = axis;

        // 标记布尔矩阵缓存失效（轴变化会改变Top-K结果）
        if (config.cachedMasks) {
            config.cachedMasks.topKMask = null;
            config.cachedMasks.finalMask = null;
        }

        console.log(`[独占模式] 保存Top-K轴: ${axis}，缓存已失效`);
    }

    // 更新按钮状态
    document.getElementById('xAxisBtn').classList.toggle('active', axis === 'x');
    document.getElementById('yAxisBtn').classList.toggle('active', axis === 'y');

    // 更新Top-K滑块的最大值
    updateTopkSliderMax();

    // 更新热力图
    if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
        updateHeatmap();
    }
}

// 【已废弃】旧的applyTopkFilter函数，保留用于兼容性检查
// 新架构使用 computeTopKMask() + applyMask() 代替
function applyTopkFilter_DEPRECATED(matrix, xLabels, yLabels, xData, yData) {
    console.warn('[已废弃] applyTopkFilter被调用，应使用新的布尔矩阵架构');
    // 临时兼容实现，直接返回原数据
    return { matrix, xLabels, yLabels, xData, yData };
}

// ============================================================================
// 相似度计算与矩阵管理
// ============================================================================

// 计算相似度矩阵
async function calculateSimilarity() {
    // 获取所有选中的X轴collections
    const xSelects = document.querySelectorAll('.x-collection-select');
    const xCollections = Array.from(xSelects)
        .map(select => select.value)
        .filter(value => value !== '');

    // 获取所有选中的Y轴collections
    const ySelects = document.querySelectorAll('.y-collection-select');
    const yCollections = Array.from(ySelects)
        .map(select => select.value)
        .filter(value => value !== '');

    const xMaxItems = parseInt(document.getElementById('xMaxItems').value) || 30;
    const yMaxItems = parseInt(document.getElementById('yMaxItems').value) || 30;

    // 先清除之前的验证错误
    clearValidationErrors();

    // 检查必要项是否填写
    if (xCollections.length === 0) {
        document.getElementById('xCollectionError').style.display = 'block';
        showError('请至少选择一个横坐标 Collection');
        return;
    }
    if (yCollections.length === 0) {
        document.getElementById('yCollectionError').style.display = 'block';
        showError('请至少选择一个纵坐标 Collection');
        return;
    }

    try {
        // 清空之前的结果
        allSimilarityResults = [];
        currentMatrixIndex = 0;

        // 计算总的请求数
        const totalRequests = xCollections.length * yCollections.length;
        let completedRequests = 0;

        showLoading(true, `正在计算相似度矩阵... (0/${totalRequests})`);

        // 遍历所有X和Y的组合
        for (let i = 0; i < xCollections.length; i++) {
            for (let j = 0; j < yCollections.length; j++) {
                const xCollection = xCollections[i];
                const yCollection = yCollections[j];

                try {
                    // 更新进度
                    completedRequests++;
                    showLoading(true, `正在计算相似度矩阵... (${completedRequests}/${totalRequests})`);

                    const data = await apiCall('/calculate', {
                        method: 'POST',
                        body: JSON.stringify({
                            x_collection: xCollection,
                            y_collection: yCollection,
                            x_max_items: xMaxItems,
                            y_max_items: yMaxItems
                        })
                    });

                    // 保存结果，并为每张图创建独立的可视化配置
                    const visualConfig = createDefaultVisualConfig(
                        data.result.x_available_fields,
                        data.result.y_available_fields
                    );

                    allSimilarityResults.push({
                        xCollection: xCollection,
                        yCollection: yCollection,
                        result: data.result,
                        matrix: data.result.matrix,
                        xData: data.result.x_data.slice(0, xMaxItems),
                        yData: data.result.y_data.slice(0, yMaxItems),
                        xAvailableFields: data.result.x_available_fields,
                        yAvailableFields: data.result.y_available_fields,
                        stats: data.result.stats,
                        visualConfig: visualConfig  // 每张图独立的可视化配置
                    });

                } catch (error) {
                    showWarning(`计算 ${xCollection} vs ${yCollection} 失败: ${error.message}`);
                }
            }
        }

        if (allSimilarityResults.length === 0) {
            throw new Error('所有相似度计算都失败了');
        }

        // *** 新架构：初始化按钮状态数组 ***
        initializeButtonStates();

        // 清空旧的状态
        selectedMatrixIndices = [];
        differenceMatrices = {};

        // 重置全局UI状态
        globalUIState.dataSource.primaryIndex = null;
        globalUIState.dataSource.subtractIndex = null;
        globalUIState.dataSource.currentMatrix = null;
        globalUIState.filters.activeFilterIndices = [];
        globalUIState.exclusiveMode.active = false;
        globalUIState.exclusiveMode.editingIndex = null;

        // *** 新架构：更新图表列表UI ***
        updateMatrixListUI();

        // 显示图表选择控制区域
        document.getElementById('matrixSelectorControl').style.display = 'block';

        showSuccess(`成功计算 ${allSimilarityResults.length} 个相似度矩阵! 请选择要显示的图表`);

    } catch (error) {
        showError('计算相似度失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 根据索引加载矩阵数据
function loadMatrixByIndex(index) {
    if (index < 0 || index >= allSimilarityResults.length) {
        return;
    }

    console.log(`[切换图表] 加载图 ${index}`);
    currentMatrixIndex = index;
    const matrixData = allSimilarityResults[index];
    console.log('[切换图表] 配置信息:', matrixData.visualConfig);

    // 更新当前数据
    currentResult = matrixData.result;
    originalMatrix = matrixData.matrix;
    originalXData = matrixData.result.x_data;
    originalYData = matrixData.result.y_data;
    xAvailableFields = matrixData.xAvailableFields;
    yAvailableFields = matrixData.yAvailableFields;

    currentXData = matrixData.xData;
    currentYData = matrixData.yData;

    // 裁剪矩阵
    filteredMatrix = originalMatrix.slice(0, currentYData.length)
        .map(row => row.slice(0, currentXData.length));

    // 更新显示字段控制器
    updateDisplayFieldControls();

    // *** 关键修改：先更新滑块范围，再应用配置值 ***
    // 1. 先更新相似度滑块的范围（根据是否差值模式设置0-1或-1-1）
    updateSimilaritySliderRange();

    // 2. 然后应用该图的可视化配置到UI（包括阈值、Top-K等）
    applyVisualConfigToUI(matrixData.visualConfig);

    // 使用该图的配置生成标签
    const xField = matrixData.visualConfig.displayFields.xField;
    const yField = matrixData.visualConfig.displayFields.yField;
    currentXLabels = generateUniqueLabels(currentXData, xField);
    currentYLabels = generateUniqueLabels(currentYData, yField);

    // 更新Top-K滑块的最大值
    updateTopkSliderMax();

    // 更新可视化 - 先创建基础热力图
    createHeatmap();

    // 然后应用该图的筛选设置（阈值和Top-K）
    updateHeatmap();

    // 显示统计信息
    showStatistics(matrixData.stats);
}

// ============================================================================
// 多图选择与切换
// ============================================================================

function updateMatrixSelector() {
    if (allSimilarityResults.length === 0) return;

    // 获取所有唯一的X和Y collections
    const xCollections = [...new Set(allSimilarityResults.map(r => r.xCollection))];
    const yCollections = [...new Set(allSimilarityResults.map(r => r.yCollection))];

    // 创建按钮表格
    const table = document.getElementById('matrixButtonTable');
    table.innerHTML = '';

    // 创建表头行
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th></th>'; // 左上角空单元格
    xCollections.forEach(xCol => {
        const th = document.createElement('th');
        th.textContent = xCol;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // 创建数据行
    yCollections.forEach(yCol => {
        const row = document.createElement('tr');

        // 第一列：Y collection名称
        const labelCell = document.createElement('td');
        labelCell.textContent = yCol;
        row.appendChild(labelCell);

        // 其他列：按钮
        xCollections.forEach(xCol => {
            const cell = document.createElement('td');

            // 查找对应的矩阵索引
            const matrixIndex = allSimilarityResults.findIndex(
                r => r.xCollection === xCol && r.yCollection === yCol
            );

            if (matrixIndex >= 0) {
                const btn = document.createElement('button');
                btn.className = 'matrix-selection-btn';
                btn.textContent = '显示';
                btn.dataset.index = matrixIndex;
                btn.onclick = function() {
                    handleMatrixButtonClick(matrixIndex);
                };
                cell.appendChild(btn);
            } else {
                cell.textContent = '-';
            }

            row.appendChild(cell);
        });

        table.appendChild(row);
    });

    // 显示控制区域
    document.getElementById('matrixSelectorControl').style.display = 'block';
}

// 新增：处理矩阵按钮点击
function handleMatrixButtonClick(clickedIndex) {
    const buttons = document.querySelectorAll('.matrix-selection-btn');

    // 情况1: 未选中任何矩阵
    if (selectedMatrixIndices.length === 0) {
        selectedMatrixIndices = [clickedIndex];
        updateButtonStates();
        loadMatrixByIndex(clickedIndex);
        updateSelectionInfo();
        return;
    }

    // 情况2: 点击的是已选中的按钮
    if (selectedMatrixIndices.includes(clickedIndex)) {
        selectedMatrixIndices = selectedMatrixIndices.filter(idx => idx !== clickedIndex);

        if (selectedMatrixIndices.length === 0) {
            // 全部取消选择
            updateButtonStates();
            updateSelectionInfo();
        } else {
            // 还有其他选中的按钮，显示剩余的
            updateButtonStates();
            loadMatrixByIndex(selectedMatrixIndices[0]);
            updateSelectionInfo();
        }
        return;
    }

    // 情况3: 已有一个选中，点击另一个
    if (selectedMatrixIndices.length === 1) {
        const firstIndex = selectedMatrixIndices[0];
        const firstMatrix = allSimilarityResults[firstIndex];
        const clickedMatrix = allSimilarityResults[clickedIndex];

        // 检查矩阵大小是否相同
        const sameSize = (
            firstMatrix.matrix.length === clickedMatrix.matrix.length &&
            firstMatrix.matrix[0].length === clickedMatrix.matrix[0].length
        );

        if (sameSize) {
            // 大小相同：显示差值
            selectedMatrixIndices = [firstIndex, clickedIndex];
            updateButtonStates();
            loadDifferenceMatrix(firstIndex, clickedIndex);
            updateSelectionInfo();
        } else {
            // 大小不同：替换选择
            selectedMatrixIndices = [clickedIndex];
            updateButtonStates();
            loadMatrixByIndex(clickedIndex);
            updateSelectionInfo();
        }
        return;
    }

    // 情况4: 已有两个选中（差值状态），点击新的
    if (selectedMatrixIndices.length === 2) {
        selectedMatrixIndices = [clickedIndex];
        updateButtonStates();
        loadMatrixByIndex(clickedIndex);
        updateSelectionInfo();
        return;
    }
}

// 新增：更新按钮状态
function updateButtonStates() {
    const buttons = document.querySelectorAll('.matrix-selection-btn');

    buttons.forEach(btn => {
        const index = parseInt(btn.dataset.index);
        btn.classList.remove('selected', 'selected-secondary', 'difference');

        if (selectedMatrixIndices.length === 1 && selectedMatrixIndices[0] === index) {
            btn.classList.add('selected');
            btn.textContent = '已选';
        } else if (selectedMatrixIndices.length === 2) {
            if (selectedMatrixIndices[0] === index) {
                btn.classList.add('selected');
                btn.textContent = '被减';
            } else if (selectedMatrixIndices[1] === index) {
                btn.classList.add('selected-secondary');
                btn.textContent = '减数';
            }
        } else {
            btn.textContent = '显示';
        }
    });
}

// 更新选择信息显示
function updateSelectionInfo() {
    const infoDiv = document.getElementById('matrixSelectionInfo');
    const textSpan = document.getElementById('selectedMatrixText');

    if (selectedMatrixIndices.length === 0) {
        infoDiv.style.display = 'none';
        return;
    }

    infoDiv.style.display = 'block';

    if (selectedMatrixIndices.length === 1) {
        const matrix = allSimilarityResults[selectedMatrixIndices[0]];
        textSpan.textContent = `当前显示: ${matrix.xCollection} vs ${matrix.yCollection}`;
    } else if (selectedMatrixIndices.length === 2) {
        const matrix1 = allSimilarityResults[selectedMatrixIndices[0]];
        const matrix2 = allSimilarityResults[selectedMatrixIndices[1]];
        textSpan.innerHTML = `差值模式: <br>(${matrix1.xCollection} vs ${matrix1.yCollection}) - (${matrix2.xCollection} vs ${matrix2.yCollection})`;
    }
}

// 加载差值矩阵
function loadDifferenceMatrix(index1, index2) {
    const cacheKey = `${index1}-${index2}`;

    // 检查缓存
    if (!differenceMatrices[cacheKey]) {
        // 计算差值矩阵
        const matrix1 = allSimilarityResults[index1].matrix;
        const matrix2 = allSimilarityResults[index2].matrix;

        const diffMatrix = matrix1.map((row, i) =>
            row.map((val, j) => val - matrix2[i][j])
        );

        differenceMatrices[cacheKey] = diffMatrix;
        // 同时缓存反向差值
        differenceMatrices[`${index2}-${index1}`] = diffMatrix.map(row =>
            row.map(val => -val)
        );
    }

    // 使用第一个矩阵的数据和字段信息
    const matrixData = allSimilarityResults[index1];
    currentResult = matrixData.result;
    originalMatrix = differenceMatrices[cacheKey];
    originalXData = matrixData.result.x_data;
    originalYData = matrixData.result.y_data;
    xAvailableFields = matrixData.xAvailableFields;
    yAvailableFields = matrixData.yAvailableFields;

    currentXData = matrixData.xData;
    currentYData = matrixData.yData;

    filteredMatrix = originalMatrix.slice(0, currentYData.length)
        .map(row => row.slice(0, currentXData.length));

    updateDisplayFieldControls();

    // *** 关键修改：差值模式使用第一张图的显示字段和筛选器 ***
    // 1. 先更新相似度滑块的范围（差值模式需要-1到1的范围）
    updateSimilaritySliderRange();

    // 2. 应用第一张图的可视化配置到UI
    applyVisualConfigToUI(matrixData.visualConfig);

    // 3. *** 差值模式：强制将阈值重置为 -1 到 1 ***
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    minSlider.value = -1;
    maxSlider.value = 1;
    minInput.value = -1;
    maxInput.value = 1;

    document.getElementById('minSimilarityValue').textContent = '-1.00';
    document.getElementById('maxSimilarityValue').textContent = '1.00';

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    track.style.left = '0%';
    track.style.width = '100%';

    console.log('[差值模式] 阈值已重置为 -1.00 到 1.00');

    // 使用第一张图的配置生成标签
    const xField = matrixData.visualConfig.displayFields.xField;
    const yField = matrixData.visualConfig.displayFields.yField;
    currentXLabels = generateUniqueLabels(currentXData, xField);
    currentYLabels = generateUniqueLabels(currentYData, yField);

    updateTopkSliderMax();

    // 更新可视化 - 先创建基础热力图
    createHeatmap();

    // 然后应用该图的筛选设置（阈值和Top-K）
    updateHeatmap();

    // 显示差值矩阵的统计信息
    showDifferenceStatistics(originalMatrix);
}

// 新增：显示差值统计信息
function showDifferenceStatistics(diffMatrix) {
    const statsSection = document.getElementById('statsSection');
    const statsGrid = document.getElementById('statsGrid');

    // 计算差值矩阵的统计信息
    let flatValues = [];
    diffMatrix.forEach(row => {
        row.forEach(val => {
            if (val !== null && val !== undefined) {
                flatValues.push(val);
            }
        });
    });

    const totalPairs = flatValues.length;
    const avgDiff = flatValues.reduce((a, b) => a + b, 0) / totalPairs;
    const maxDiff = Math.max(...flatValues);
    const minDiff = Math.min(...flatValues);

    const currentDisplayCount = getCurrentDisplayCount();

    statsGrid.innerHTML = `
        <div class="stat-item">
            <div class="value">${totalPairs}</div>
            <div class="label">总对比数</div>
        </div>
        <div class="stat-item">
            <div class="value" id="currentDisplayCount">${currentDisplayCount}</div>
            <div class="label">当前显示对比数</div>
        </div>
        <div class="stat-item">
            <div class="value">${avgDiff.toFixed(3)}</div>
            <div class="label">平均差值</div>
        </div>
        <div class="stat-item">
            <div class="value">${maxDiff.toFixed(3)}</div>
            <div class="label">最大差值</div>
        </div>
        <div class="stat-item">
            <div class="value">${minDiff.toFixed(3)}</div>
            <div class="label">最小差值</div>
        </div>
        <div class="stat-item">
            <div class="value">差值模式</div>
            <div class="label">当前模式</div>
        </div>
    `;

    statsSection.style.display = 'block';
    operationsSection.style.display = 'block';
}



// ============================================================================
// 可视化更新与渲染
// ============================================================================

// 更新Top-K滑块的最大值
function updateTopkSliderMax() {
    if (!filteredMatrix) return;

    const maxX = filteredMatrix[0] ? filteredMatrix[0].length : 0;
    const maxY = filteredMatrix.length;

    // 根据当前选择的轴确定最大值
    const maxTopk = currentTopkAxis === 'x' ? maxX : maxY;

    const topkSlider = document.getElementById('topkSlider');
    topkSlider.max = maxTopk;

    // 如果当前值超过最大值，重置为最大值
    if (parseInt(topkSlider.value) > maxTopk) {
        topkSlider.value = maxTopk;
        updateTopkDisplay();
    }

    // 更新增减按钮状态
    updateTopkButtons();
}

// 更新Top-K增减按钮状态
function updateTopkButtons() {
    // 移除所有禁用逻辑，按钮始终可用
    const decBtn = document.getElementById('topkDecBtn');
    const incBtn = document.getElementById('topkIncBtn');

    decBtn.disabled = false;
    incBtn.disabled = false;
}

// 调整Top-K值
function adjustTopk(delta) {
    const topkSlider = document.getElementById('topkSlider');
    const currentValue = parseInt(topkSlider.value);
    const minValue = parseInt(topkSlider.min);
    const maxValue = parseInt(topkSlider.max);
    const newValue = Math.max(minValue, Math.min(maxValue, currentValue + delta));
    if (newValue !== currentValue) {
        topkSlider.value = newValue;
        updateTopkDisplay();
        // 实时更新热力图
        if (filteredMatrix) {
            updateHeatmap();
        }
    }

    // 确保按钮状态正确更新
    updateTopkButtons();
}

// 计算当前显示的对比数 - 新架构：使用布尔矩阵
function getCurrentDisplayCount() {
    if (!filteredMatrix) return 0;

    // 获取当前应用的最终遮罩
    let finalMask = null;

    // 情况1: 独占模式
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        const editingIndex = globalUIState.exclusiveMode.editingIndex;
        const config = allSimilarityResults[editingIndex].visualConfig;
        finalMask = config.cachedMasks?.finalMask || computeFinalMaskForMatrix(editingIndex);
    }
    // 情况2: 应用筛选器模式
    else if (globalUIState.filters.activeFilterIndices.length > 0) {
        const masks = globalUIState.filters.activeFilterIndices.map(index => {
            const config = allSimilarityResults[index].visualConfig;
            return config.cachedMasks?.finalMask || computeFinalMaskForMatrix(index);
        }).filter(mask => mask !== null);

        if (masks.length > 0) {
            finalMask = combineWithOR(masks);
        }
    }
    // 情况3: 默认模式
    else {
        const minSim = parseFloat(document.getElementById('minSimilaritySlider').value);
        const maxSim = parseFloat(document.getElementById('maxSimilaritySlider').value);
        const topK = parseInt(document.getElementById('topkSlider').value);
        const axis = currentTopkAxis;

        const thresholdMask = computeThresholdMask(filteredMatrix, minSim, maxSim);
        const topKMask = computeTopKMask(filteredMatrix, topK, axis);
        finalMask = combineWithAND(thresholdMask, topKMask);
    }

    // 统计true的数量
    if (!finalMask) return 0;

    let count = 0;
    finalMask.forEach(row => {
        row.forEach(val => {
            if (val === true) {
                count++;
            }
        });
    });
    return count;
}

// 更新热力图（实时响应控制变化）- 新架构：使用布尔矩阵
function updateHeatmap() {
    if (!filteredMatrix) {
        return;
    }

    console.log('[updateHeatmap] 开始更新 - 使用新布尔矩阵架构');

    // 保存当前的缩放和选中状态
    let currentLayout = null;
    const heatmapDiv = document.getElementById('heatmap');
    if (heatmapDiv && heatmapDiv.layout) {
        currentLayout = {
            xaxis: {
                range: heatmapDiv.layout.xaxis.range,
                autorange: heatmapDiv.layout.xaxis.autorange
            },
            yaxis: {
                range: heatmapDiv.layout.yaxis.range,
                autorange: heatmapDiv.layout.yaxis.autorange
            }
        };
    }

    const sortOrder = document.getElementById('sortOrder').value;

    // ========== 新架构核心逻辑 ==========
    let finalMask = null;

    // 情况1: 独占模式 - 使用单张图的布尔矩阵
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        const editingIndex = globalUIState.exclusiveMode.editingIndex;
        console.log(`[updateHeatmap] 独占模式 - 使用图 ${editingIndex} 的遮罩`);

        // 计算该图的最终遮罩（阈值 AND Top-K）
        finalMask = computeFinalMaskForMatrix(editingIndex);
    }
    // 情况2: 应用筛选器模式 - OR合并多个图的布尔矩阵
    else if (globalUIState.filters.activeFilterIndices.length > 0) {
        console.log(`[updateHeatmap] 应用筛选器模式 - 合并 ${globalUIState.filters.activeFilterIndices.length} 个图的遮罩`);

        // 收集所有活动筛选器的最终遮罩
        const masks = globalUIState.filters.activeFilterIndices.map(index =>
            computeFinalMaskForMatrix(index)
        ).filter(mask => mask !== null);

        // OR合并
        if (masks.length > 0) {
            finalMask = combineWithOR(masks);
        }
    }
    // 情况3: 默认模式 - 使用当前UI控件的值（兼容旧逻辑）
    else {
        console.log('[updateHeatmap] 默认模式 - 基于当前UI控件计算遮罩');

        const minSim = parseFloat(document.getElementById('minSimilaritySlider').value);
        const maxSim = parseFloat(document.getElementById('maxSimilaritySlider').value);
        const topK = parseInt(document.getElementById('topkSlider').value);
        const axis = currentTopkAxis;

        // 直接计算布尔矩阵
        const thresholdMask = computeThresholdMask(filteredMatrix, minSim, maxSim);
        const topKMask = computeTopKMask(filteredMatrix, topK, axis);
        finalMask = combineWithAND(thresholdMask, topKMask);
    }

    // 应用遮罩到原始数据
    let displayMatrix = filteredMatrix;
    if (finalMask) {
        displayMatrix = applyMask(filteredMatrix, finalMask);
    }

    let displayXLabels = [...currentXLabels];
    let displayYLabels = [...currentYLabels];
    let displayXData = [...currentXData];
    let displayYData = [...currentYData];

    // 应用排序
    if (sortOrder !== 'none') {
        const sortedData = applySorting(displayMatrix, displayXLabels, displayYLabels, displayXData, displayYData, sortOrder);
        displayMatrix = sortedData.matrix;
        displayXLabels = sortedData.xLabels;
        displayYLabels = sortedData.yLabels;
    }

    // 更新热力图数据，但保持当前的缩放状态
    updateHeatmapData(displayMatrix, displayXLabels, displayYLabels, currentLayout);

    // 更新统计信息中的当前显示对比数
    updateCurrentDisplayStat();
}

// 更新热力图数据但保持缩放状态
function updateHeatmapData(matrix, xLabels, yLabels, preserveLayout = null) {
    // 获取当前模式的相似度范围
    const range = getCurrentSimilarityRange();

    const trace = {
        z: matrix,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: colorSchemes[currentColorScheme],
        hoverongaps: false,
        hovertemplate: '<b>%{y}</b><br>' +
                      '<b>%{x}</b><br>' +
                      '<b>相似度: %{z:.4f}</b>' +
                      '<extra></extra>',
        colorbar: {
            title: isDifferenceMode() ? '差值' : '相似度',
            titleside: 'right',
            tickmode: 'linear',
            tick0: range.min,
            dtick: isDifferenceMode() ? 0.2 : 0.1
        },
        showscale: true,
        zmin: range.min,  // 固定最小值
        zmax: range.max   // 固定最大值
    };

    // 如果有保存的布局状态，则应用它
    if (preserveLayout) {
        const currentHeatmapDiv = document.getElementById('heatmap');

        // 使用 Plotly.restyle 只更新数据，不影响布局
        Plotly.restyle('heatmap', {
            z: [matrix],
            x: [xLabels],
            y: [yLabels],
            colorscale: [colorSchemes[currentColorScheme]],
            zmin: [range.min],  // 添加这行
            zmax: [range.max]   // 添加这行
        });

        // 如果需要恢复特定的缩放范围，使用 relayout
        if (preserveLayout.xaxis.range || preserveLayout.yaxis.range) {
            const layoutUpdate = {};

            if (preserveLayout.xaxis.range) {
                layoutUpdate['xaxis.range'] = preserveLayout.xaxis.range;
                layoutUpdate['xaxis.autorange'] = false;
            }

            if (preserveLayout.yaxis.range) {
                layoutUpdate['yaxis.range'] = preserveLayout.yaxis.range;
                layoutUpdate['yaxis.autorange'] = false;
            }

            if (Object.keys(layoutUpdate).length > 0) {
                Plotly.relayout('heatmap', layoutUpdate);
            }
        }
    } else {
        // 如果没有保存的布局状态，使用完整的重新绘制
        createHeatmap(matrix, xLabels, yLabels);
    }
}

// 更新相似度滑块的范围
// resetToRange: true=重置为范围边界(range.min, range.max), false=保持当前值
function updateSimilaritySliderRange(resetToRange = false) {
    const range = getCurrentSimilarityRange();
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    // 更新滑块范围属性
    minSlider.min = range.min;
    minSlider.max = range.max;
    minSlider.step = 0.01;

    maxSlider.min = range.min;
    maxSlider.max = range.max;
    maxSlider.step = 0.01;

    // 更新输入框范围属性
    minInput.min = range.min;
    minInput.max = range.max;

    maxInput.min = range.min;
    maxInput.max = range.max;

    // 根据参数决定是重置还是保持当前值
    if (resetToRange) {
        // 重置为范围边界
        minSlider.value = range.min;
        maxSlider.value = range.max;
    }
    // 如果不重置，滑块值保持不变（会由后续的 applyVisualConfigToUI 设置）

    minInput.value = minSlider.value;
    maxInput.value = maxSlider.value;

    // 更新显示
    document.getElementById('minSimilarityValue').textContent = parseFloat(minSlider.value).toFixed(2);
    document.getElementById('maxSimilarityValue').textContent = parseFloat(maxSlider.value).toFixed(2);

    // 更新track位置
    const track = document.getElementById('similarityTrack');
    const percent1 = ((parseFloat(minSlider.value) - range.min) / (range.max - range.min)) * 100;
    const percent2 = ((parseFloat(maxSlider.value) - range.min) / (range.max - range.min)) * 100;
    track.style.left = percent1 + '%';
    track.style.width = (percent2 - percent1) + '%';
}


// 更新当前显示对比数统计
function updateCurrentDisplayStat() {
    const currentDisplayElement = document.getElementById('currentDisplayCount');
    if (currentDisplayElement) {
        const currentCount = getCurrentDisplayCount();
        currentDisplayElement.textContent = currentCount;
    }
}

// 应用排序逻辑，使用唯一标签
function applySorting(matrix, xLabels, yLabels, xData, yData, sortOrder) {
    let indices = [];

    switch (sortOrder) {
        case 'asc':
        case 'desc':
            // 按相似度排序：计算每个位置的平均相似度
            for (let i = 0; i < matrix.length; i++) {
                for (let j = 0; j < matrix[i].length; j++) {
                    if (matrix[i][j] !== null) {
                        indices.push({
                            x: j, y: i,
                            similarity: matrix[i][j],
                            xLabel: xLabels[j],
                            yLabel: yLabels[i]
                        });
                    }
                }
            }
            indices.sort((a, b) => sortOrder === 'asc' ?
                a.similarity - b.similarity : b.similarity - a.similarity);
            break;

        case 'x_asc':
        case 'x_desc':
            // 按X轴标签排序 - 获取当前显示字段
            const xDisplayField = document.getElementById('xDisplayField').value;
            const xIndexOrder = xData.map((item, index) => ({
                value: xDisplayField === 'order_id' ? item.order_id : String(item[xDisplayField] || ''),
                index: index,
                originalIndex: index  // 保存原始索引，用于处理重复值
            })).sort((a, b) => {
                const aVal = String(a.value);
                const bVal = String(b.value);

                // 如果值相同，按照原始索引排序保持稳定性
                if (aVal === bVal) {
                    return a.originalIndex - b.originalIndex;
                }

                return sortOrder === 'x_asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            });

            // 重新生成唯一标签，保持排序后的顺序
            const sortedXData = xIndexOrder.map(x => xData[x.index]);
            const newXLabels = generateUniqueLabels(sortedXData, xDisplayField);
            const newMatrix = matrix.map(row =>
                xIndexOrder.map(x => row[x.index])
            );

            return { matrix: newMatrix, xLabels: newXLabels, yLabels: [...yLabels] };

        case 'y_asc':
        case 'y_desc':
            // 按Y轴标签排序 - 获取当前显示字段
            const yDisplayField = document.getElementById('yDisplayField').value;
            const yIndexOrder = yData.map((item, index) => ({
                value: yDisplayField === 'order_id' ? item.order_id : String(item[yDisplayField] || ''),
                index: index,
                originalIndex: index  // 保存原始索引，用于处理重复值
            })).sort((a, b) => {
                const aVal = String(a.value);
                const bVal = String(b.value);

                // 如果值相同，按照原始索引排序保持稳定性
                if (aVal === bVal) {
                    return a.originalIndex - b.originalIndex;
                }

                return sortOrder === 'y_asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            });

            // 重新生成唯一标签，保持排序后的顺序
            const sortedYData = yIndexOrder.map(y => yData[y.index]);
            const newYLabels = generateUniqueLabels(sortedYData, yDisplayField);
            const sortedMatrix = yIndexOrder.map(y => matrix[y.index]);

            return { matrix: sortedMatrix, xLabels: [...xLabels], yLabels: newYLabels };
    }

    return { matrix, xLabels, yLabels };
}
// 判断当前是否为差值模式（兼容旧代码）
function isDifferenceMode() {
    return isInDifferenceMode();
}

// 获取当前模式下的相似度范围
function getCurrentSimilarityRange() {
    return isDifferenceMode() ? { min: -1, max: 1 } : { min: 0, max: 1 };
}


// 创建热力图
function createHeatmap(matrix = filteredMatrix, xLabels = currentXLabels, yLabels = currentYLabels) {
    if (!matrix) {
        showError('没有相似度数据');
        return;
    }

    // 获取当前模式的相似度范围
    const range = getCurrentSimilarityRange();

    const trace = {
        z: matrix,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: colorSchemes[currentColorScheme],
        hoverongaps: false,
        hovertemplate: '<b>%{y}</b><br>' +
                      '<b>%{x}</b><br>' +
                      '<b>相似度: %{z:.4f}</b>' +
                      '<extra></extra>',
        colorbar: {
            title: isDifferenceMode() ? '差值' : '相似度',
            titleside: 'right',
            tickmode: 'linear',
            tick0: range.min,
            dtick: isDifferenceMode() ? 0.2 : 0.1
        },
        showscale: true,
        zmin: range.min,  // 固定最小值
        zmax: range.max   // 固定最大值
    };

    // 获取容器实际尺寸
    const container = document.querySelector('.heatmap-container');
    const containerRect = container.getBoundingClientRect();

    // 计算可用空间（减去padding和controls的高度）
    const availableWidth = containerRect.width - 24; // 减去左右padding
    const controlsHeight = document.querySelector('.heatmap-controls').offsetHeight;
    const availableHeight = containerRect.height - controlsHeight - 24; // 减去controls高度和padding

    // 动态生成坐标轴标题
    const matrixData = allSimilarityResults[currentMatrixIndex];
    const xCollectionName = matrixData ? matrixData.xCollection : '';
    const yCollectionName = matrixData ? matrixData.yCollection : '';

    const xDisplayField = document.getElementById('xDisplayField').value;
    const yDisplayField = document.getElementById('yDisplayField').value;

    const xFieldName = xDisplayField === 'order_id' ? '顺序ID' : xDisplayField;
    const yFieldName = yDisplayField === 'order_id' ? '顺序ID' : yDisplayField;

    const layout = {
        title: {
            text: '相似度热力图',
            font: { size: 14, color: '#404040' },
            x: 0.5
        },
        xaxis: {
            title: `${xCollectionName} (${xFieldName})`,
            tickangle: -45,
            side: 'bottom',
            tickfont: { size: 9 },
            titlefont: { color: '#404040' }
        },
        yaxis: {
            title: `${yCollectionName} (${yFieldName})`,
            autorange: 'reversed',
            tickfont: { size: 9 },
            titlefont: { color: '#404040' }
        },
        margin: { l: 70, r: 50, t: 50, b: 80 },
        width: availableWidth,
        height: availableHeight,
        autosize: false,
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    };

    const config = {
        responsive: false, // 改为false，使用手动控制尺寸
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true
    };

    // 使用 Plotly.newPlot 重新创建图表
    Plotly.newPlot('heatmap', [trace], layout, config);
}

// 显示统计信息
function showStatistics(stats) {
    const statsSection = document.getElementById('statsSection');
    const statsGrid = document.getElementById('statsGrid');

    const currentDisplayCount = getCurrentDisplayCount();

    statsGrid.innerHTML = `
        <div class="stat-item">
            <div class="value">${stats.total_pairs}</div>
            <div class="label">总对比数</div>
        </div>
        <div class="stat-item">
            <div class="value" id="currentDisplayCount">${currentDisplayCount}</div>
            <div class="label">当前显示对比数</div>
        </div>
        <div class="stat-item">
            <div class="value">${stats.avg_similarity.toFixed(3)}</div>
            <div class="label">平均相似度</div>
        </div>
        <div class="stat-item">
            <div class="value">${stats.max_similarity.toFixed(3)}</div>
            <div class="label">最高相似度</div>
        </div>
        <div class="stat-item">
            <div class="value">${stats.min_similarity.toFixed(3)}</div>
            <div class="label">最低相似度</div>
        </div>
        <div class="stat-item">
            <div class="value">${(stats.compute_time / 1000).toFixed(1)}s</div>
            <div class="label">计算耗时</div>
        </div>
    `;

    statsSection.style.display = 'block';
    operationsSection.style.display = 'block';
}

// ============================================================================
// UI控件初始化
// ============================================================================

// 初始化双滑块
function initRangeSlider() {
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');
    const minValue = document.getElementById('minSimilarityValue');
    const maxValue = document.getElementById('maxSimilarityValue');
    const track = document.getElementById('similarityTrack');

    function updateTrack() {
        const min = parseFloat(minSlider.value);
        const max = parseFloat(maxSlider.value);
        const percent1 = (min - parseFloat(minSlider.min)) / (parseFloat(minSlider.max) - parseFloat(minSlider.min)) * 100;
        const percent2 = (max - parseFloat(maxSlider.min)) / (parseFloat(maxSlider.max) - parseFloat(maxSlider.min)) * 100;

        track.style.left = percent1 + '%';
        track.style.width = (percent2 - percent1) + '%';

        minValue.textContent = min.toFixed(2);
        maxValue.textContent = max.toFixed(2);
        minInput.value = min;
        maxInput.value = max;

        // 更新全局筛选器状态
        globalUIState.filters.uiState.similarityRange.min = min;
        globalUIState.filters.uiState.similarityRange.max = max;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.similarityRange.min = min;
            config.similarityRange.max = max;

            // 标记布尔矩阵缓存失效
            if (config.cachedMasks) {
                config.cachedMasks.thresholdMask = null;
                config.cachedMasks.finalMask = null;
            }

            console.log(`[独占模式] 保存阈值范围: ${min.toFixed(2)} - ${max.toFixed(2)}，缓存已失效`);
        }

        // 实时更新热力图
        if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
            updateHeatmap();
        }
    }

    minSlider.addEventListener('input', function() {
        if (parseFloat(this.value) > parseFloat(maxSlider.value)) {
            this.value = maxSlider.value;
        }
        updateTrack();
    });

    maxSlider.addEventListener('input', function() {
        if (parseFloat(this.value) < parseFloat(minSlider.value)) {
            this.value = minSlider.value;
        }
        updateTrack();
    });

    minInput.addEventListener('change', function() {
        const val = Math.max(parseFloat(minSlider.min), Math.min(parseFloat(minSlider.max), parseFloat(this.value) || 0));
        minSlider.value = val;
        updateTrack();
    });

    maxInput.addEventListener('change', function() {
        const val = Math.max(parseFloat(maxSlider.min), Math.min(parseFloat(maxSlider.max), parseFloat(this.value) || 1));
        maxSlider.value = val;
        updateTrack();
    });

    updateTrack();
}

// 初始化Top-K滑块
function initTopkSlider() {
    const topkSlider = document.getElementById('topkSlider');
    function updateTopkDisplayInternal() {
        const topkSlider = document.getElementById('topkSlider');
        const topkValue = parseInt(topkSlider.value);
        document.getElementById('topkValue').textContent = topkValue;
        document.getElementById('topkStatus').textContent = topkValue === 0 ? '显示全部' : `显示Top-${topkValue}`;

        // 更新全局筛选器状态
        globalUIState.filters.uiState.topK.value = topkValue;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.filters.topK.value = topkValue;

            // 标记布尔矩阵缓存失效
            if (config.cachedMasks) {
                config.cachedMasks.topKMask = null;
                config.cachedMasks.finalMask = null;
            }

            console.log(`[独占模式] 保存Top-K值: ${topkValue}，缓存已失效`);
        }

        // 更新按钮状态
        updateTopkButtons();
    }
    // 添加input事件监听器，实现拖动时实时更新热力图
    topkSlider.addEventListener('input', function() {
        updateTopkDisplayInternal();
        // 实时更新热力图
        if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
            updateHeatmap();
        }
    });

    updateTopkDisplayInternal();
}

// 更新Top-K显示（供外部调用）
function updateTopkDisplay() {
    const topkSlider = document.getElementById('topkSlider');
    const topkValue = parseInt(topkSlider.value);
    document.getElementById('topkValue').textContent = topkValue;
    document.getElementById('topkStatus').textContent = topkValue === 0 ? '显示全部' : `显示Top-${topkValue}`;
}

// 初始化颜色方案选择器
function initColorSchemeSelector() {
    const colorBtns = document.querySelectorAll('.colorscheme-btn');

    colorBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // 移除所有active类
            colorBtns.forEach(b => b.classList.remove('active'));
            // 添加当前active类
            this.classList.add('active');

            currentColorScheme = this.dataset.scheme;

            // 重新绘制热力图
            if (filteredMatrix) {
                updateHeatmap();
            }
        });
    });
}

// 初始化排序选择器
function initSortSelector() {
    document.getElementById('sortOrder').addEventListener('change', function() {
        const sortOrder = this.value;

        // 更新全局排序状态
        globalUIState.sorting.order = sortOrder;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.sorting.order = sortOrder;
            console.log(`[独占模式] 保存排序方式: ${sortOrder}`);
        }

        if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
            updateHeatmap();
        }
    });
}

// ============================================================================
// 颜色映射与样式
// ============================================================================

// 根据相似度值和当前颜色方案计算RGB颜色
function getSimilarityColor(similarity, colorScheme) {
    if (similarity === null || similarity === undefined) {
        return { r: 255, g: 255, b: 255 }; // 白色背景表示无数据
    }

    // 将相似度值映射到0-1范围
    const normalizedValue = Math.max(0, Math.min(1, similarity));

    let r, g, b;

    switch (colorScheme) {
        case 'viridis':
            // Viridis颜色映射近似
            if (normalizedValue < 0.25) {
                const t = normalizedValue / 0.25;
                r = Math.round(68 + (59 - 68) * t);
                g = Math.round(1 + (82 - 1) * t);
                b = Math.round(84 + (139 - 84) * t);
            } else if (normalizedValue < 0.5) {
                const t = (normalizedValue - 0.25) / 0.25;
                r = Math.round(59 + (33 - 59) * t);
                g = Math.round(82 + (144 - 82) * t);
                b = Math.round(139 + (140 - 139) * t);
            } else if (normalizedValue < 0.75) {
                const t = (normalizedValue - 0.5) / 0.25;
                r = Math.round(33 + (94 - 33) * t);
                g = Math.round(144 + (201 - 144) * t);
                b = Math.round(140 + (97 - 140) * t);
            } else {
                const t = (normalizedValue - 0.75) / 0.25;
                r = Math.round(94 + (253 - 94) * t);
                g = Math.round(201 + (231 - 201) * t);
                b = Math.round(97 + (37 - 97) * t);
            }
            break;

        case 'plasma':
            // Plasma颜色映射近似
            if (normalizedValue < 0.33) {
                const t = normalizedValue / 0.33;
                r = Math.round(13 + (123 - 13) * t);
                g = Math.round(8 + (15 - 8) * t);
                b = Math.round(135 + (200 - 135) * t);
            } else if (normalizedValue < 0.66) {
                const t = (normalizedValue - 0.33) / 0.33;
                r = Math.round(123 + (202 - 123) * t);
                g = Math.round(15 + (15 - 15) * t);
                b = Math.round(200 + (161 - 200) * t);
            } else {
                const t = (normalizedValue - 0.66) / 0.34;
                r = Math.round(202 + (240 - 202) * t);
                g = Math.round(15 + (249 - 15) * t);
                b = Math.round(161 + (33 - 161) * t);
            }
            break;

        case 'cool':
            // Cool颜色映射
            r = Math.round(61 + (31 - 61) * normalizedValue);
            g = Math.round(72 + (119 - 72) * normalizedValue);
            b = Math.round(73 + (180 - 73) * normalizedValue);
            break;

        case 'hot':
            // Hot颜色映射
            if (normalizedValue < 0.33) {
                const t = normalizedValue / 0.33;
                r = Math.round(0 + (255 - 0) * t);
                g = 0;
                b = 0;
            } else if (normalizedValue < 0.66) {
                const t = (normalizedValue - 0.33) / 0.33;
                r = 255;
                g = Math.round(0 + (255 - 0) * t);
                b = 0;
            } else {
                const t = (normalizedValue - 0.66) / 0.34;
                r = 255;
                g = 255;
                b = Math.round(0 + (255 - 0) * t);
            }
            break;


        default:
            // 默认灰度映射
            const gray = Math.round(255 * (1 - normalizedValue));
            r = gray;
            g = gray;
            b = gray;
    }

    return { r: Math.max(0, Math.min(255, r)),
             g: Math.max(0, Math.min(255, g)),
             b: Math.max(0, Math.min(255, b)) };
}

// ============================================================================
// 导出功能
// ============================================================================

// 导出Excel功能
async function exportToExcel() {
    if (!filteredMatrix) {
        showError('没有可导出的数据，请先计算相似度');
        return;
    }

    try {
        // 显示导出状态
        const exportBtn = document.getElementById('exportBtn');
        const originalText = exportBtn.textContent;
        exportBtn.textContent = '导出中...';
        exportBtn.disabled = true;

        // 获取当前显示的数据
        const minSim = parseFloat(document.getElementById('minSimilaritySlider').value);
        const maxSim = parseFloat(document.getElementById('maxSimilaritySlider').value);
        const sortOrder = document.getElementById('sortOrder').value;

        // 应用过滤和排序
        let thresholdMatrix = filteredMatrix.map(row =>
            row.map(val => (val >= minSim && val <= maxSim) ? val : null)
        );

        let topkResult = applyTopkFilter(thresholdMatrix, currentXLabels, currentYLabels, currentXData, currentYData);
        let displayMatrix = topkResult.matrix;
        let displayXLabels = [...topkResult.xLabels];
        let displayYLabels = [...topkResult.yLabels];
        let displayXData = [...topkResult.xData];
        let displayYData = [...topkResult.yData];

        // 应用排序
        if (sortOrder !== 'none') {
            const sortedData = applySorting(displayMatrix, displayXLabels, displayYLabels, displayXData, displayYData, sortOrder);
            displayMatrix = sortedData.matrix;
            displayXLabels = sortedData.xLabels;
            displayYLabels = sortedData.yLabels;
        }

        // 创建Excel工作簿
        const workbook = new ExcelJS.Workbook();
        const worksheet = workbook.addWorksheet('相似度矩阵');

        // 获取当前显示字段信息
        const xCollection = document.getElementById('xCollection').value;
        const yCollection = document.getElementById('yCollection').value;
        const xDisplayField = document.getElementById('xDisplayField').value;
        const yDisplayField = document.getElementById('yDisplayField').value;

        const xFieldName = xDisplayField === 'order_id' ? '顺序ID' : xDisplayField;
        const yFieldName = yDisplayField === 'order_id' ? '顺序ID' : yDisplayField;

        // 设置表头：第一行
        const headerRow = [''];
        displayXLabels.forEach(label => {
            // 移除零宽字符用于Excel显示
            const cleanLabel = label.replace(/[\u200B\u00A0]/g, '');
            headerRow.push(cleanLabel);
        });
        worksheet.addRow(headerRow);

        // 填充数据行
        displayMatrix.forEach((row, rowIndex) => {
            const dataRow = [];
            // 第一列：Y轴标签
            const cleanYLabel = displayYLabels[rowIndex].replace(/[\u200B\u00A0]/g, '');
            dataRow.push(cleanYLabel);

            // 其余列：相似度数据
            row.forEach((similarity) => {
                dataRow.push(similarity);
            });
            worksheet.addRow(dataRow);
        });

        // 设置第一行和第一列为冻结窗格
        worksheet.views = [{
            state: 'frozen',
            xSplit: 1,  // 冻结第一列
            ySplit: 1   // 冻结第一行
        }];

        // 设置表头样式
        const headerRowObj = worksheet.getRow(1);
        headerRowObj.eachCell((cell, colNumber) => {
            cell.fill = {
                type: 'pattern',
                pattern: 'solid',
                fgColor: { argb: 'FF0D6EFD' }
            };
            cell.font = {
                color: { argb: 'FFFFFFFF' },
                bold: true
            };
            cell.alignment = {
                horizontal: 'center',
                vertical: 'middle'
            };
        });

        // 设置第一列样式（Y轴标签）
        for (let rowIndex = 2; rowIndex <= displayMatrix.length + 1; rowIndex++) {
            const cell = worksheet.getCell(rowIndex, 1);
            cell.fill = {
                type: 'pattern',
                pattern: 'solid',
                fgColor: { argb: 'FF0D6EFD' }
            };
            cell.font = {
                color: { argb: 'FFFFFFFF' },
                bold: true
            };
            cell.alignment = {
                horizontal: 'center',
                vertical: 'middle'
            };
        }

        // 为相似度数据单元格设置颜色和格式
        for (let rowIndex = 0; rowIndex < displayMatrix.length; rowIndex++) {
            for (let colIndex = 0; colIndex < displayMatrix[rowIndex].length; colIndex++) {
                const similarity = displayMatrix[rowIndex][colIndex];
                const excelRow = rowIndex + 2; // Excel行从2开始（第1行是表头）
                const excelCol = colIndex + 2; // Excel列从2开始（第1列是Y轴标签）

                const cell = worksheet.getCell(excelRow, excelCol);

                if (similarity !== null && similarity !== undefined) {
                    // 设置数值和格式
                    cell.value = similarity;
                    cell.numFmt = '0.0000';

                    // 根据相似度和当前颜色方案设置背景色
                    const color = getSimilarityColor(similarity, currentColorScheme);
                    const argbColor = `FF${color.r.toString(16).padStart(2, '0')}${color.g.toString(16).padStart(2, '0')}${color.b.toString(16).padStart(2, '0')}`;

                    cell.fill = {
                        type: 'pattern',
                        pattern: 'solid',
                        fgColor: { argb: argbColor }
                    };

                    // 根据背景色亮度调整文字颜色
                    const brightness = (color.r * 299 + color.g * 587 + color.b * 114) / 1000;
                    cell.font = {
                        color: { argb: brightness > 128 ? 'FF000000' : 'FFFFFFFF' },
                        size: 10
                    };
                } else {
                    // 无数据的单元格
                    cell.value = '';
                    cell.fill = {
                        type: 'pattern',
                        pattern: 'solid',
                        fgColor: { argb: 'FFFFFFFF' }
                    };
                }

                // 设置单元格边框和对齐方式
                cell.border = {
                    top: { style: 'thin' },
                    left: { style: 'thin' },
                    bottom: { style: 'thin' },
                    right: { style: 'thin' }
                };
                cell.alignment = {
                    horizontal: 'center',
                    vertical: 'middle'
                };
            }
        }

        // 自动调整列宽
        worksheet.columns.forEach((column) => {
            column.width = 15;
        });

        // 设置第一列稍微宽一些以容纳标签
        worksheet.getColumn(1).width = 20;

        // 生成文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `相似度矩阵_${xCollection}_vs_${yCollection}_${timestamp}.xlsx`;

        // 导出Excel文件
        const buffer = await workbook.xlsx.writeBuffer();
        const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showSuccess(`Excel文件导出成功：${filename}`);

    } catch (error) {
        console.error('导出Excel时出错:', error);
        showError('导出Excel失败: ' + error.message);
    } finally {
        // 恢复按钮状态
        const exportBtn = document.getElementById('exportBtn');
        exportBtn.textContent = '导出Excel';  // 直接设置为固定文本，更可靠
        exportBtn.disabled = false;
    }
}


// 导出JSON功能
async function exportToJSON() {
    if (!allSimilarityResults || allSimilarityResults.length === 0) {
        showError('没有可导出的数据，请先计算相似度');
        return;
    }

    try {
        // 显示导出状态
        const exportBtn = document.getElementById('exportJsonBtn');
        const originalText = exportBtn.textContent;
        exportBtn.textContent = '导出中...';
        exportBtn.disabled = true;

        // 获取所有选中的X和Y collections
        const xSelects = document.querySelectorAll('.x-collection-select');
        const xCollections = Array.from(xSelects)
            .map(select => select.value)
            .filter(value => value !== '');

        const ySelects = document.querySelectorAll('.y-collection-select');
        const yCollections = Array.from(ySelects)
            .map(select => select.value)
            .filter(value => value !== '');

        // 构建完整的导出数据结构
        const exportData = {
            version: "1.0",
            timestamp: new Date().toISOString(),
            config: {
                xCollections: xCollections,
                yCollections: yCollections,
                xMaxItems: parseInt(document.getElementById('xMaxItems').value) || 30,
                yMaxItems: parseInt(document.getElementById('yMaxItems').value) || 30
            },
            results: allSimilarityResults.map(result => ({
                xCollection: result.xCollection,
                yCollection: result.yCollection,
                matrix: result.matrix,
                xData: result.xData,
                yData: result.yData,
                xAvailableFields: result.xAvailableFields,
                yAvailableFields: result.yAvailableFields,
                stats: result.stats
            })),
            displaySettings: {
                xDisplayField: document.getElementById('xDisplayField').value,
                yDisplayField: document.getElementById('yDisplayField').value,
                minSimilarity: parseFloat(document.getElementById('minSimilaritySlider').value),
                maxSimilarity: parseFloat(document.getElementById('maxSimilaritySlider').value),
                topkValue: parseInt(document.getElementById('topkSlider').value),
                topkAxis: currentTopkAxis,
                sortOrder: document.getElementById('sortOrder').value,
                colorScheme: currentColorScheme
            },
            matrixSelection: {
                selectedMatrixIndices: selectedMatrixIndices,
                currentMatrixIndex: currentMatrixIndex
            }
        };

        // 转换为JSON字符串
        const jsonString = JSON.stringify(exportData, null, 2);

        // 生成文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `相似度分析_完整_${timestamp}.json`;

        // 创建Blob并下载
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showSuccess(`JSON文件导出成功：${filename}`);

    } catch (error) {
        console.error('导出JSON时出错:', error);
        showError('导出JSON失败: ' + error.message);
    } finally {
        // 恢复按钮状态
        const exportBtn = document.getElementById('exportJsonBtn');
        exportBtn.textContent = '导出JSON';
        exportBtn.disabled = false;
    }
}

// ============================================================================
// 事件监听器
// ============================================================================

// 监听窗口大小变化，重新调整图表大小
window.addEventListener('resize', function() {
    if (document.getElementById('heatmap').innerHTML && filteredMatrix) {
        // 延迟执行以确保容器尺寸已更新
        setTimeout(() => {
            const container = document.querySelector('.heatmap-container');
            const containerRect = container.getBoundingClientRect();
            const controlsHeight = document.querySelector('.heatmap-controls').offsetHeight;
            const availableWidth = containerRect.width - 24;
            const availableHeight = containerRect.height - controlsHeight - 24;

            Plotly.relayout('heatmap', {
                width: availableWidth,
                height: availableHeight
            });
        }, 150);
    }
});

// ============================================================================
// 新架构：核心状态管理函数
// ============================================================================

/**
 * 初始化按钮状态数组
 */
function initializeButtonStates() {
    matrixButtonStates = allSimilarityResults.map((_, index) => ({
        index: index,
        applyData: false,
        applyFilter: false,
        exclusive: false
    }));
}

/**
 * 切换"应用数据"按钮
 * @param {number} index - 图表索引
 */
function toggleApplyDataButton(index) {
    console.log(`[应用数据] 切换按钮 ${index}`);

    const currentState = matrixButtonStates[index].applyData;

    // 如果当前是独占模式，先退出
    if (globalUIState.exclusiveMode.active) {
        exitExclusiveMode();
    }

    if (currentState) {
        // 当前已启用，要关闭
        if (globalUIState.dataSource.primaryIndex === index) {
            // 关闭主数据源
            globalUIState.dataSource.primaryIndex = null;
            globalUIState.dataSource.currentMatrix = null;
            matrixButtonStates[index].applyData = false;

            // 如果有减数，提升为主数据源
            if (globalUIState.dataSource.subtractIndex !== null) {
                const subtractIdx = globalUIState.dataSource.subtractIndex;
                globalUIState.dataSource.primaryIndex = subtractIdx;
                globalUIState.dataSource.subtractIndex = null;
                loadDataFromMatrix(subtractIdx, false);
            }
        } else if (globalUIState.dataSource.subtractIndex === index) {
            // 关闭减数
            globalUIState.dataSource.subtractIndex = null;
            matrixButtonStates[index].applyData = false;
            // 重新加载主数据源（退出差值模式）
            loadDataFromMatrix(globalUIState.dataSource.primaryIndex, false);
        }
    } else {
        // 当前未启用，要开启
        if (globalUIState.dataSource.primaryIndex === null) {
            // 没有主数据源，设为主数据源
            globalUIState.dataSource.primaryIndex = index;
            matrixButtonStates[index].applyData = true;
            loadDataFromMatrix(index, false);
        } else if (globalUIState.dataSource.subtractIndex === null) {
            // 已有主数据源，没有减数，设为减数（差值模式）
            globalUIState.dataSource.subtractIndex = index;
            matrixButtonStates[index].applyData = true;
            loadDifferenceData(globalUIState.dataSource.primaryIndex, index);
        } else {
            // 已有主数据源和减数，替换减数
            const oldSubtract = globalUIState.dataSource.subtractIndex;
            matrixButtonStates[oldSubtract].applyData = false;
            globalUIState.dataSource.subtractIndex = index;
            matrixButtonStates[index].applyData = true;
            loadDifferenceData(globalUIState.dataSource.primaryIndex, index);
        }
    }

    updateMatrixListUI();
    updateHeatmapFromGlobalState();
}

/**
 * 从矩阵加载数据到全局状态
 * @param {number} index - 图表索引
 * @param {boolean} isDifference - 是否为差值模式
 */
function loadDataFromMatrix(index, isDifference = false) {
    console.log(`[加载数据] 从图 ${index} 加载，差值模式: ${isDifference}`);

    const matrixData = allSimilarityResults[index];

    // 加载原始数据
    globalUIState.dataSource.currentMatrix = matrixData.matrix;
    globalUIState.dataSource.currentXData = matrixData.xData;
    globalUIState.dataSource.currentYData = matrixData.yData;
    globalUIState.dataSource.xAvailableFields = matrixData.xAvailableFields;
    globalUIState.dataSource.yAvailableFields = matrixData.yAvailableFields;

    // 更新显示字段（使用该图的配置）
    const defaultXField = matrixData.visualConfig.displayFields.xField;
    const defaultYField = matrixData.visualConfig.displayFields.yField;

    globalUIState.displayFields.xField = defaultXField;
    globalUIState.displayFields.yField = defaultYField;

    // 生成标签
    globalUIState.dataSource.currentXLabels = generateUniqueLabels(
        globalUIState.dataSource.currentXData,
        globalUIState.displayFields.xField
    );
    globalUIState.dataSource.currentYLabels = generateUniqueLabels(
        globalUIState.dataSource.currentYData,
        globalUIState.displayFields.yField
    );

    // 更新显示字段选择器
    updateDisplayFieldSelectorsFromGlobalState();
}

/**
 * 加载差值数据
 * @param {number} primaryIndex - 主数据源索引
 * @param {number} subtractIndex - 减数索引
 */
function loadDifferenceData(primaryIndex, subtractIndex) {
    console.log(`[差值模式] 主图: ${primaryIndex}, 减数图: ${subtractIndex}`);

    const cacheKey = `${primaryIndex}-${subtractIndex}`;

    // 检查缓存
    if (!differenceMatrices[cacheKey]) {
        const matrix1 = allSimilarityResults[primaryIndex].matrix;
        const matrix2 = allSimilarityResults[subtractIndex].matrix;

        // 检查矩阵大小是否一致
        if (matrix1.length !== matrix2.length || matrix1[0].length !== matrix2[0].length) {
            showError('无法计算差值：两个矩阵大小不一致');
            return;
        }

        const diffMatrix = matrix1.map((row, i) =>
            row.map((val, j) => val - matrix2[i][j])
        );

        differenceMatrices[cacheKey] = diffMatrix;
    }

    // 使用主数据源的元数据，但矩阵是差值
    const primaryData = allSimilarityResults[primaryIndex];

    globalUIState.dataSource.currentMatrix = differenceMatrices[cacheKey];
    globalUIState.dataSource.currentXData = primaryData.xData;
    globalUIState.dataSource.currentYData = primaryData.yData;
    globalUIState.dataSource.xAvailableFields = primaryData.xAvailableFields;
    globalUIState.dataSource.yAvailableFields = primaryData.yAvailableFields;

    // 显示字段不变（使用主图的）
    // 只在首次加载主图时设置，差值模式下保持

    // 生成标签
    globalUIState.dataSource.currentXLabels = generateUniqueLabels(
        globalUIState.dataSource.currentXData,
        globalUIState.displayFields.xField
    );
    globalUIState.dataSource.currentYLabels = generateUniqueLabels(
        globalUIState.dataSource.currentYData,
        globalUIState.displayFields.yField
    );

    // *** 差值模式：将阈值重置为 -1 到 1 ***
    globalUIState.filters.uiState.similarityRange.min = -1;
    globalUIState.filters.uiState.similarityRange.max = 1;

    // 更新UI滑块
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    minSlider.value = -1;
    maxSlider.value = 1;
    minInput.value = -1;
    maxInput.value = 1;

    document.getElementById('minSimilarityValue').textContent = '-1.00';
    document.getElementById('maxSimilarityValue').textContent = '1.00';

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    track.style.left = '0%';
    track.style.width = '100%';

    console.log('[差值模式] 阈值已重置为 -1.00 到 1.00');
}

/**
 * 切换"应用筛选器"按钮
 * @param {number} index - 图表索引
 */
function toggleApplyFilterButton(index) {
    console.log(`[应用筛选器] 切换按钮 ${index}`);

    // 如果当前是独占模式，先退出
    if (globalUIState.exclusiveMode.active) {
        exitExclusiveMode();
    }

    const currentState = matrixButtonStates[index].applyFilter;

    if (currentState) {
        // 当前已启用，要关闭
        matrixButtonStates[index].applyFilter = false;
        globalUIState.filters.activeFilterIndices = globalUIState.filters.activeFilterIndices.filter(i => i !== index);
    } else {
        // 当前未启用，要开启
        matrixButtonStates[index].applyFilter = true;
        globalUIState.filters.activeFilterIndices.push(index);
    }

    // 合并筛选器（或逻辑）
    mergeFilters();

    updateMatrixListUI();
    updateHeatmapFromGlobalState();
}

/**
 * 合并多个筛选器（或逻辑）- 新架构：基于布尔矩阵
 *
 * 注意：这个函数现在主要用于更新UI显示
 * 实际的OR合并在updateHeatmap()中通过combineWithOR()完成
 */
function mergeFilters() {
    console.log(`[合并筛选器] 活动筛选器数量: ${globalUIState.filters.activeFilterIndices.length}`);

    if (globalUIState.filters.activeFilterIndices.length === 0) {
        // 没有活动筛选器，使用默认值
        globalUIState.filters.uiState.similarityRange = { min: 0, max: 1 };
        globalUIState.filters.uiState.topK = { value: 0, axis: 'x' };
    } else if (globalUIState.filters.activeFilterIndices.length === 1) {
        // 只有一个筛选器，直接使用
        const index = globalUIState.filters.activeFilterIndices[0];
        const config = allSimilarityResults[index].visualConfig;
        globalUIState.filters.uiState.similarityRange = { ...config.similarityRange };
        globalUIState.filters.uiState.topK = { ...config.filters.topK };
    } else {
        // 多个筛选器：为了UI显示友好，显示合并后的范围
        // 但实际筛选逻辑在updateHeatmap()中基于布尔矩阵OR运算完成
        let minRange = 1, maxRange = 0;
        let maxTopK = 0;
        let topKAxis = 'x';

        globalUIState.filters.activeFilterIndices.forEach(index => {
            const config = allSimilarityResults[index].visualConfig;
            minRange = Math.min(minRange, config.similarityRange.min);
            maxRange = Math.max(maxRange, config.similarityRange.max);
            if (config.filters.topK.value > maxTopK) {
                maxTopK = config.filters.topK.value;
                topKAxis = config.filters.topK.axis;
            }
        });

        globalUIState.filters.uiState.similarityRange = { min: minRange, max: maxRange };
        globalUIState.filters.uiState.topK = { value: maxTopK, axis: topKAxis };

        console.log(`[合并筛选器] UI显示范围: 阈值 ${minRange.toFixed(2)}-${maxRange.toFixed(2)}, Top-K ${maxTopK}`);
    }

    // 更新UI控件（仅用于显示）
    applyFilterStateToUI();
}

/**
 * 切换"独占模式"按钮
 * @param {number} index - 图表索引
 */
function toggleExclusiveModeButton(index) {
    console.log(`[独占模式] 切换按钮 ${index}`);

    const currentState = matrixButtonStates[index].exclusive;

    if (currentState) {
        // 当前已是独占模式，退出
        exitExclusiveMode();
    } else {
        // 进入独占模式
        enterExclusiveMode(index);
    }

    updateMatrixListUI();
}

/**
 * 进入独占模式
 * @param {number} index - 图表索引
 */
function enterExclusiveMode(index) {
    console.log(`[独占模式] 进入，编辑图 ${index}`);

    // 关闭所有其他按钮
    matrixButtonStates.forEach((state, i) => {
        if (i !== index) {
            state.applyData = false;
            state.applyFilter = false;
            state.exclusive = false;
        }
    });

    // 启用当前图的两个按钮
    matrixButtonStates[index].applyData = true;
    matrixButtonStates[index].applyFilter = true;
    matrixButtonStates[index].exclusive = true;

    // 设置全局状态
    globalUIState.exclusiveMode.active = true;
    globalUIState.exclusiveMode.editingIndex = index;

    // 清空数据源和筛选器
    globalUIState.dataSource.primaryIndex = index;
    globalUIState.dataSource.subtractIndex = null;
    globalUIState.filters.activeFilterIndices = [index];

    // 加载该图的所有配置
    loadDataFromMatrix(index, false);

    // 应用该图的筛选器配置
    const config = allSimilarityResults[index].visualConfig;
    globalUIState.filters.uiState.similarityRange = { ...config.similarityRange };
    globalUIState.filters.uiState.topK = { ...config.filters.topK };
    globalUIState.sorting.order = config.sorting.order;

    // 更新UI
    applyFilterStateToUI();
    applySortingStateToUI();

    // 显示提示
    showInfo('已进入独占编辑模式，您的修改将保存到此图的配置中', 3000);

    updateHeatmapFromGlobalState();
}

/**
 * 退出独占模式
 */
function exitExclusiveMode() {
    if (!globalUIState.exclusiveMode.active) return;

    const editingIndex = globalUIState.exclusiveMode.editingIndex;
    console.log(`[独占模式] 退出，保存图 ${editingIndex} 的配置`);

    // 保存当前UI状态到图表配置
    if (editingIndex !== null && allSimilarityResults[editingIndex]) {
        const config = allSimilarityResults[editingIndex].visualConfig;

        // 保存显示字段
        config.displayFields.xField = globalUIState.displayFields.xField;
        config.displayFields.yField = globalUIState.displayFields.yField;

        // 保存筛选器
        config.similarityRange = { ...globalUIState.filters.uiState.similarityRange };
        config.filters.topK = { ...globalUIState.filters.uiState.topK };

        // 保存排序
        config.sorting.order = globalUIState.sorting.order;

        console.log(`[独占模式] 已保存配置:`, config);
    }

    // 重置独占模式状态
    globalUIState.exclusiveMode.active = false;
    globalUIState.exclusiveMode.editingIndex = null;

    matrixButtonStates.forEach(state => {
        state.exclusive = false;
    });

    showSuccess('已保存配置并退出独占模式', 2000);
}

/**
 * 应用筛选器状态到UI控件
 */
function applyFilterStateToUI() {
    const range = globalUIState.filters.uiState.similarityRange;
    const topK = globalUIState.filters.uiState.topK;

    // 更新相似度范围
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    minSlider.value = range.min;
    maxSlider.value = range.max;
    minInput.value = range.min;
    maxInput.value = range.max;

    document.getElementById('minSimilarityValue').textContent = range.min.toFixed(2);
    document.getElementById('maxSimilarityValue').textContent = range.max.toFixed(2);

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    const percent1 = ((range.min - parseFloat(minSlider.min)) / (parseFloat(minSlider.max) - parseFloat(minSlider.min))) * 100;
    const percent2 = ((range.max - parseFloat(maxSlider.min)) / (parseFloat(maxSlider.max) - parseFloat(maxSlider.min))) * 100;
    track.style.left = percent1 + '%';
    track.style.width = (percent2 - percent1) + '%';

    // 更新Top-K
    document.getElementById('topkSlider').value = topK.value;
    currentTopkAxis = topK.axis;
    updateTopkDisplay();

    // 更新轴选择按钮
    document.getElementById('xAxisBtn').classList.toggle('active', topK.axis === 'x');
    document.getElementById('yAxisBtn').classList.toggle('active', topK.axis === 'y');
}

/**
 * 应用排序状态到UI控件
 */
function applySortingStateToUI() {
    document.getElementById('sortOrder').value = globalUIState.sorting.order;
}

/**
 * 更新显示字段选择器
 */
function updateDisplayFieldSelectorsFromGlobalState() {
    const xFields = globalUIState.dataSource.xAvailableFields;
    const yFields = globalUIState.dataSource.yAvailableFields;

    // 显示控件
    document.getElementById('displayFieldControls').style.display = 'block';
    document.getElementById('yDisplayFieldControls').style.display = 'block';

    // 更新X轴选择器
    const xSelect = document.getElementById('xDisplayField');
    xSelect.innerHTML = '';
    xFields.forEach(field => {
        const option = new Option(
            field === 'order_id' ? '顺序ID' : field,
            field
        );
        xSelect.add(option);
    });
    xSelect.value = globalUIState.displayFields.xField;

    // 更新Y轴选择器
    const ySelect = document.getElementById('yDisplayField');
    ySelect.innerHTML = '';
    yFields.forEach(field => {
        const option = new Option(
            field === 'order_id' ? '顺序ID' : field,
            field
        );
        ySelect.add(option);
    });
    ySelect.value = globalUIState.displayFields.yField;
}

/**
 * 从全局状态更新热力图
 */
function updateHeatmapFromGlobalState() {
    if (!globalUIState.dataSource.currentMatrix) {
        console.log('[热力图] 没有数据源，跳过更新');
        return;
    }

    console.log('[热力图] 从全局状态更新');

    // 使用全局状态的数据
    filteredMatrix = globalUIState.dataSource.currentMatrix;
    currentXData = globalUIState.dataSource.currentXData;
    currentYData = globalUIState.dataSource.currentYData;
    currentXLabels = globalUIState.dataSource.currentXLabels;
    currentYLabels = globalUIState.dataSource.currentYLabels;
    xAvailableFields = globalUIState.dataSource.xAvailableFields;
    yAvailableFields = globalUIState.dataSource.yAvailableFields;

    // 更新相似度滑块范围（差值模式需要 -1 到 1）
    updateSimilaritySliderRangeForGlobalState();

    // 创建或更新热力图
    if (document.getElementById('heatmap').innerHTML === '') {
        createHeatmap();
    } else {
        updateHeatmap();
    }

    // 显示统计信息
    if (globalUIState.dataSource.primaryIndex !== null) {
        const stats = allSimilarityResults[globalUIState.dataSource.primaryIndex].stats;
        if (globalUIState.dataSource.subtractIndex !== null) {
            showDifferenceStatistics(filteredMatrix);
        } else {
            showStatistics(stats);
        }
    }

    // 显示操作区域
    document.getElementById('statsSection').style.display = 'block';
    document.getElementById('operationsSection').style.display = 'block';
}

/**
 * 更新相似度滑块范围（根据是否差值模式）
 */
function updateSimilaritySliderRangeForGlobalState() {
    const isDiff = globalUIState.dataSource.subtractIndex !== null;
    const range = isDiff ? { min: -1, max: 1 } : { min: 0, max: 1 };

    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    minSlider.min = range.min;
    minSlider.max = range.max;
    maxSlider.min = range.min;
    maxSlider.max = range.max;
    minInput.min = range.min;
    minInput.max = range.max;
    maxInput.min = range.min;
    maxInput.max = range.max;
}

/**
 * 检查是否为差值模式
 */
function isInDifferenceMode() {
    return globalUIState.dataSource.subtractIndex !== null;
}

/**
 * 更新图表列表UI（新架构：竖排列表 + 三按钮）
 */
function updateMatrixListUI() {
    const container = document.getElementById('matrixButtonTable');
    if (!container) return;

    // 清空现有内容
    container.innerHTML = '';

    // 生成竖排列表
    allSimilarityResults.forEach((result, index) => {
        const state = matrixButtonStates[index];

        // 创建行容器
        const row = document.createElement('div');
        row.className = 'matrix-list-item';
        row.dataset.index = index;

        // 图表标题
        const title = document.createElement('div');
        title.className = 'matrix-item-title';
        title.textContent = `${result.xCollection} vs ${result.yCollection}`;
        title.title = `点击查看详情`;

        // 按钮容器
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'matrix-item-buttons';

        // 按钮1：应用数据
        const dataBtn = document.createElement('button');
        dataBtn.className = 'matrix-control-btn btn-apply-data';
        dataBtn.textContent = '应用数据';
        dataBtn.title = '使用该图的相似度数值';
        if (state.applyData) dataBtn.classList.add('active');
        dataBtn.onclick = () => toggleApplyDataButton(index);

        // 按钮2：应用筛选器
        const filterBtn = document.createElement('button');
        filterBtn.className = 'matrix-control-btn btn-apply-filter';
        filterBtn.textContent = '应用筛选器';
        filterBtn.title = '使用该图的筛选条件（支持多选-或逻辑）';
        if (state.applyFilter) filterBtn.classList.add('active');
        filterBtn.onclick = () => toggleApplyFilterButton(index);

        // 按钮3：独占模式
        const exclusiveBtn = document.createElement('button');
        exclusiveBtn.className = 'matrix-control-btn btn-exclusive';
        exclusiveBtn.textContent = '独占模式';
        exclusiveBtn.title = '独占编辑此图的配置';
        if (state.exclusive) exclusiveBtn.classList.add('active');
        exclusiveBtn.onclick = () => toggleExclusiveModeButton(index);

        // 状态指示器
        const statusIndicator = document.createElement('div');
        statusIndicator.className = 'matrix-item-status';

        let statusText = '';
        if (globalUIState.dataSource.primaryIndex === index && globalUIState.dataSource.subtractIndex === null) {
            statusText = '主数据源';
            statusIndicator.classList.add('status-primary');
        } else if (globalUIState.dataSource.primaryIndex === index && globalUIState.dataSource.subtractIndex !== null) {
            statusText = '被减数';
            statusIndicator.classList.add('status-primary');
        } else if (globalUIState.dataSource.subtractIndex === index) {
            statusText = '减数';
            statusIndicator.classList.add('status-subtract');
        }

        if (state.exclusive) {
            statusText = '🔒 独占编辑中';
            statusIndicator.classList.add('status-exclusive');
        }

        statusIndicator.textContent = statusText;

        // 组装
        buttonsContainer.appendChild(dataBtn);
        buttonsContainer.appendChild(filterBtn);
        buttonsContainer.appendChild(exclusiveBtn);

        row.appendChild(title);
        row.appendChild(statusIndicator);
        row.appendChild(buttonsContainer);

        container.appendChild(row);
    });
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    initRangeSlider();
    initTopkSlider();
    initColorSchemeSelector();
    initSortSelector();
    loadCollections(); // 自动加载Collections
    // 为 collection 选择框添加事件委托
    document.getElementById('xCollectionContainer').addEventListener('change', function(e) {
        if (e.target.classList.contains('x-collection-select')) {
            document.getElementById('xCollectionError').style.display = 'none';
        }
    });

    document.getElementById('yCollectionContainer').addEventListener('change', function(e) {
        if (e.target.classList.contains('y-collection-select')) {
            document.getElementById('yCollectionError').style.display = 'none';
        }
    });
});
