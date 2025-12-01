// 知识库测试页面 JavaScript

// =====
// 性能优化: 防抖计时器
// =====
let heatmapUpdateTimer = null;
let statsUpdateTimer = null;
const DEBOUNCE_DELAY = 5; // 5ms防抖延迟

// =====
// 动态侧边栏配置与逻辑
// =====

// 侧边栏内容数据 - 定义6个功能模块
const sidebarData = {
    dataSource: {
        title: '数据源配置',
        titleButton: '<button class="title-icon-btn" onclick="loadCollections()" title="刷新Collections"><i class="bi bi-arrow-clockwise"></i></button>',
        templateId: 'tpl-dataSource'
    },
    operations: {
        title: '结果操作',
        templateId: 'tpl-operations'
    },
    chartControl: {
        title: '图表选择与控制',
        titleButton: '<button class="title-icon-btn" onclick="triggerImportJSON()" title="导入数据"><i class="bi bi-file-earmark-arrow-up"></i></button>',
        templateId: 'tpl-chartControl'
    },
    statistics: {
        title: '统计信息',
        templateId: 'tpl-statistics'
    },
    filters: {
        title: '筛选器控制',
        templateId: 'tpl-filters'
    }
};

// 当前激活的按钮 - 分为上下两个区域
let activeButtons = {
    left: {
        top: null,
        bottom: null
    },
    right: {
        top: null,
        bottom: null
    }
};

// 保存侧边栏的自定义宽度
let sidebarWidths = {
    left: null,
    right: null
};

// 保存上下分区的高度比例(存储上部分的像素高度)
let splitHeights = {
    left: null,
    right: null
};

// 从localStorage加载保存的宽度和高度
try {
    const savedWidths = localStorage.getItem('sidebarWidths_testPage');
    if (savedWidths) {
        sidebarWidths = JSON.parse(savedWidths);
    }
    const savedHeights = localStorage.getItem('splitHeights_testPage');
    if (savedHeights) {
        splitHeights = JSON.parse(savedHeights);
    }
} catch (e) {
    console.error('加载侧边栏配置失败', e);
}

// 拖动状态
let dragState = {
    isDragging: false,
    draggedButton: null,
    dragGhost: null,
    dropIndicator: null,
    startX: 0,
    startY: 0,
    dragThreshold: 5,
    hasMoved: false
};

// 预加载所有侧边栏面板内容到统一的隐藏容器
function preloadAllPanels() {
    console.log('[侧边栏] 预加载所有面板内容');

    // 创建或获取全局隐藏容器
    let globalPanelContainer = document.getElementById('globalPanelContainer');
    if (!globalPanelContainer) {
        globalPanelContainer = document.createElement('div');
        globalPanelContainer.id = 'globalPanelContainer';
        globalPanelContainer.style.display = 'none';
        document.body.appendChild(globalPanelContainer);
    }

    // 为每个面板创建DOM，全部存储在隐藏容器中
    Object.keys(sidebarData).forEach(panelName => {
        const data = sidebarData[panelName];
        if (!data) {
            console.warn(`面板 "${panelName}" 配置不存在`);
            return;
        }

        const panelDiv = document.createElement('div');
        panelDiv.id = `panel-${panelName}`;
        panelDiv.className = 'sidebar-panel';
        panelDiv.style.display = 'none';

        // 构建标题HTML,如果有titleButton则添加
        const titleHTML = data.titleButton
            ? `<div class="panel-title-wrapper">
                   <h3 class="panel-title">${data.title}</h3>
                   ${data.titleButton}
               </div>`
            : `<h3 style="font-size: 14px; font-weight: 600; margin-bottom: 15px; color: #404040; padding-bottom: 10px; border-bottom: 2px solid #6666FF;">${data.title}</h3>`;

        // 获取模板内容
        let contentHTML = '';
        if (data.templateId) {
            const tpl = document.getElementById(data.templateId);
            if (tpl) {
                contentHTML = tpl.innerHTML;
            } else {
                console.error(`Template #${data.templateId} not found for panel ${panelName}`);
            }
        } else if (data.content) {
            contentHTML = data.content;
        }

        panelDiv.innerHTML = titleHTML + contentHTML;

        globalPanelContainer.appendChild(panelDiv);
    });

    // 一次性初始化所有面板的事件处理器
    console.log('[侧边栏] 初始化所有面板的事件处理器');
    Object.keys(sidebarData).forEach(panelName => {
        rebindEventHandlers(panelName);
    });
}

// 更新侧边栏内容 - 动态移动面板DOM到目标位置
function updateSidebarContent(side, position, panel) {
    const contentId = `${side}Content${position.charAt(0).toUpperCase() + position.slice(1)}`;
    const content = document.getElementById(contentId);

    if (!content) {
        console.warn(`容器 #${contentId} 不存在`);
        return;
    }

    // 隐藏该区域当前的所有面板
    content.querySelectorAll('.sidebar-panel').forEach(p => {
        p.style.display = 'none';
    });

    // 从全局容器或其他位置获取面板
    const panelDiv = document.getElementById(`panel-${panel}`);
    if (!panelDiv) {
        console.warn(`面板 #panel-${panel} 不存在`);
        return;
    }

    // 将面板移动到目标容器（如果不在的话）
    if (panelDiv.parentElement !== content) {
        content.appendChild(panelDiv);
    }

    // 显示面板
    panelDiv.style.display = 'block';
    console.log(`[侧边栏] 面板 ${panel} 移动到 ${contentId} 并显示`);

    // 如果是筛选器面板,在显示后更新相似度滑块轨道
    if (panel === 'filters' && typeof window.updateSimilarityTrack === 'function') {
        setTimeout(() => {
            window.updateSimilarityTrack();
        }, 50);
    }
}

// 绑定面板事件处理器（只在预加载时调用一次）
function rebindEventHandlers(panel) {
    console.log(`[事件绑定] 初始化面板: ${panel}`);

    // 根据不同的面板绑定相应的事件处理器
    switch (panel) {
        case 'filters':
            // 初始化筛选器控件
            if (typeof initRangeSlider === 'function') {
                initRangeSlider();
            }
            if (typeof initTopkSlider === 'function') {
                initTopkSlider();
            }
            // 初始化显示字段选择器（现在在筛选器面板中）
            if (typeof initDisplayFieldSelectors === 'function') {
                initDisplayFieldSelectors();
            }
            break;

        case 'dataSource':
            // 加载collections数据
            if (typeof loadCollections === 'function') {
                loadCollections();
            }

            // 为 collection 选择框添加事件委托
            const xContainer = document.getElementById('xCollectionContainer');
            const yContainer = document.getElementById('yCollectionContainer');

            if (xContainer) {
                xContainer.addEventListener('change', function (e) {
                    if (e.target.classList.contains('x-collection-select')) {
                        const errorEl = document.getElementById('xCollectionError');
                        if (errorEl) errorEl.style.display = 'none';
                    }
                });
            }

            if (yContainer) {
                yContainer.addEventListener('change', function (e) {
                    if (e.target.classList.contains('y-collection-select')) {
                        const errorEl = document.getElementById('yCollectionError');
                        if (errorEl) errorEl.style.display = 'none';
                    }
                });
            }
            break;

        case 'chartControl':
            // 图表控制面板已简化，不需要额外的事件绑定
            break;

        case 'statistics':
            // 统计信息是动态填充的,不需要特别的事件绑定
            break;

        case 'operations':
            // 导出操作的事件已经通过全局绑定
            break;
    }
}

// 更新侧边栏整体状态
function updateSidebarState(side) {
    const sidebar = document.getElementById(`${side}Sidebar`);
    const contentTop = document.getElementById(`${side}ContentTop`);
    const contentBottom = document.getElementById(`${side}ContentBottom`);
    const verticalHandle = sidebar.querySelector('.vertical-resize-handle');

    const hasTop = activeButtons[side].top !== null;
    const hasBottom = activeButtons[side].bottom !== null;

    // 记录当前状态以检测变化
    const wasOpen = sidebar.classList.contains('open');
    const shouldBeOpen = hasTop || hasBottom;
    const stateChanged = wasOpen !== shouldBeOpen;

    if (!hasTop && !hasBottom) {
        sidebar.classList.remove('open');
        sidebar.style.width = '0';
        contentTop.classList.remove('active', 'full-height');
        contentBottom.classList.remove('active', 'full-height');
        verticalHandle.classList.remove('active');
        // 只在侧边栏状态真正改变时调整热力图大小
        if (stateChanged) {
            resizeHeatmap();
        }
        return;
    }

    sidebar.classList.add('open');
    if (sidebarWidths[side]) {
        sidebar.style.width = sidebarWidths[side] + 'px';
    } else {
        sidebar.style.width = '320px';
    }
    // 只在侧边栏状态真正改变时调整热力图大小
    if (stateChanged) {
        resizeHeatmap();
    }

    if (hasTop && hasBottom) {
        contentTop.classList.add('active');
        contentTop.classList.remove('full-height');
        contentBottom.classList.add('active');
        contentBottom.classList.remove('full-height');
        verticalHandle.classList.add('active');

        if (splitHeights[side]) {
            contentTop.style.height = splitHeights[side] + 'px';
            contentBottom.style.flex = '1';
        } else {
            contentTop.style.height = '50%';
            contentBottom.style.flex = '1';
        }
    } else if (hasTop) {
        contentTop.classList.add('active', 'full-height');
        contentBottom.classList.remove('active', 'full-height');
        verticalHandle.classList.remove('active');
        contentTop.style.height = '';
        contentTop.style.flex = '';
        contentBottom.style.flex = '';
    } else {
        contentTop.classList.remove('active', 'full-height');
        contentBottom.classList.add('active', 'full-height');
        verticalHandle.classList.remove('active');
        contentBottom.style.height = '';
        contentTop.style.flex = '';
        contentBottom.style.flex = '';
    }
}

// 初始化侧边栏
function initializeSidebar() {
    // *** 关键改动：首先预加载所有面板内容 ***
    preloadAllPanels();

    const buttons = document.querySelectorAll('.icon-btn');

    buttons.forEach(button => {
        button.addEventListener('mousedown', function (e) {
            dragState.draggedButton = this;
            dragState.startX = e.clientX;
            dragState.startY = e.clientY;
            dragState.hasMoved = false;
            e.preventDefault();
        });

        button.addEventListener('click', function () {
            if (dragState.hasMoved) {
                return;
            }

            const side = this.dataset.side;
            const position = this.dataset.position;
            const panel = this.dataset.panel;

            if (activeButtons[side][position] === this) {
                this.classList.remove('active');
                activeButtons[side][position] = null;
            } else {
                if (activeButtons[side][position]) {
                    activeButtons[side][position].classList.remove('active');
                }

                this.classList.add('active');
                activeButtons[side][position] = this;

                updateSidebarContent(side, position, panel);
            }

            updateSidebarState(side);
        });
    });

    document.addEventListener('mousemove', function (e) {
        if (!dragState.draggedButton) return;

        const deltaX = Math.abs(e.clientX - dragState.startX);
        const deltaY = Math.abs(e.clientY - dragState.startY);

        if (!dragState.isDragging && (deltaX > dragState.dragThreshold || deltaY > dragState.dragThreshold)) {
            startDragging(e);
        }

        if (dragState.isDragging) {
            updateDragGhost(e);
            updateDropIndicator(e);
        }
    });

    document.addEventListener('mouseup', function (e) {
        if (dragState.isDragging) {
            finishDragging(e);
        } else {
            dragState.draggedButton = null;
        }
    });

    // 初始化调整大小逻辑
    initializeResizeHandles();

    // 触发默认激活的按钮
    triggerDefaultActiveButtons();
}

// 初始化调整大小手柄
function initializeResizeHandles() {
    const resizeHandles = document.querySelectorAll('.resize-handle');

    resizeHandles.forEach(handle => {
        let isResizing = false;
        let sidebar = null;
        let startX = 0;
        let startWidth = 0;
        let resizeTimer = null;
        const side = handle.dataset.side;

        handle.addEventListener('mousedown', function (e) {
            sidebar = document.getElementById(`${side}Sidebar`);
            if (!sidebar.classList.contains('open')) return;

            isResizing = true;
            startX = e.clientX;
            startWidth = sidebar.offsetWidth;
            handle.classList.add('dragging');

            e.preventDefault();
        });

        document.addEventListener('mousemove', function (e) {
            if (!isResizing) return;

            let newWidth;
            if (side === 'left') {
                newWidth = startWidth + (e.clientX - startX);
            } else {
                newWidth = startWidth - (e.clientX - startX);
            }

            newWidth = Math.max(250, Math.min(newWidth, 600));

            sidebar.style.width = newWidth + 'px';

            // 节流resize调用，避免过于频繁
            if (resizeTimer) {
                clearTimeout(resizeTimer);
            }
            resizeTimer = setTimeout(() => {
                resizeHeatmap();
            }, 50);
        });

        document.addEventListener('mouseup', function () {
            if (isResizing) {
                isResizing = false;
                handle.classList.remove('dragging');

                if (sidebar) {
                    sidebarWidths[side] = sidebar.offsetWidth;

                    try {
                        localStorage.setItem('sidebarWidths_testPage', JSON.stringify(sidebarWidths));
                    } catch (e) {
                        console.error('保存侧边栏宽度失败', e);
                    }

                    // 调整完成后，resize热力图
                    resizeHeatmap();
                }
            }
        });
    });

    // 垂直调整大小
    const verticalResizeHandles = document.querySelectorAll('.vertical-resize-handle');

    verticalResizeHandles.forEach(handle => {
        let isResizing = false;
        let contentTop = null;
        let startY = 0;
        let startHeight = 0;
        const side = handle.dataset.side;

        handle.addEventListener('mousedown', function (e) {
            if (!handle.classList.contains('active')) return;

            contentTop = document.getElementById(`${side}ContentTop`);

            isResizing = true;
            startY = e.clientY;
            startHeight = contentTop.offsetHeight;
            handle.classList.add('dragging');

            e.preventDefault();
        });

        document.addEventListener('mousemove', function (e) {
            if (!isResizing) return;

            const deltaY = e.clientY - startY;
            const newHeight = startHeight + deltaY;

            const sidebar = document.getElementById(`${side}Sidebar`);
            const totalHeight = sidebar.offsetHeight;

            const minHeight = 100;
            const maxHeight = totalHeight - minHeight - 4;

            const clampedHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));

            contentTop.style.height = clampedHeight + 'px';
        });

        document.addEventListener('mouseup', function () {
            if (isResizing) {
                isResizing = false;
                handle.classList.remove('dragging');

                if (contentTop) {
                    splitHeights[side] = contentTop.offsetHeight;

                    try {
                        localStorage.setItem('splitHeights_testPage', JSON.stringify(splitHeights));
                    } catch (e) {
                        console.error('保存分区高度失败', e);
                    }
                }
            }
        });
    });
}

// 触发默认激活的按钮
function triggerDefaultActiveButtons() {
    // 左上默认激活: dataSource
    const leftTopBtn = document.querySelector('[data-side="left"][data-position="top"][data-panel="dataSource"]');
    if (leftTopBtn) {
        leftTopBtn.click();
    }

    // 左下默认激活: chartControl
    const leftBottomBtn = document.querySelector('[data-side="left"][data-position="bottom"][data-panel="chartControl"]');
    if (leftBottomBtn) {
        leftBottomBtn.click();
    }
}

// =====
// 拖动功能相关函数
// =====

// 开始拖动
function startDragging(e) {
    dragState.isDragging = true;
    dragState.hasMoved = true;

    // 添加拖动样式
    dragState.draggedButton.classList.add('dragging');

    // 创建幽灵图标
    const ghost = document.createElement('div');
    ghost.className = 'drag-ghost';
    ghost.innerHTML = dragState.draggedButton.innerHTML;
    document.body.appendChild(ghost);
    dragState.dragGhost = ghost;

    // 创建放置指示器
    const indicator = document.createElement('div');
    indicator.className = 'drop-indicator';
    dragState.dropIndicator = indicator;

    updateDragGhost(e);
}

// 更新幽灵图标位置
function updateDragGhost(e) {
    if (dragState.dragGhost) {
        dragState.dragGhost.style.left = e.clientX + 'px';
        dragState.dragGhost.style.top = e.clientY + 'px';
    }
}

// 更新放置指示器位置
function updateDropIndicator(e) {
    if (!dragState.dropIndicator) return;

    // 获取鼠标位置下的目标位置
    const dropTarget = getDropTarget(e);

    // 移除现有的指示器
    if (dragState.dropIndicator.parentNode) {
        dragState.dropIndicator.remove();
    }

    if (dropTarget) {
        if (dropTarget.isEmpty) {
            // 空分组: 将指示器放在分组中间
            dropTarget.element.appendChild(dragState.dropIndicator);
        } else {
            // 有按钮的分组: 在目标按钮前后插入指示器
            if (dropTarget.before) {
                dropTarget.element.parentNode.insertBefore(dragState.dropIndicator, dropTarget.element);
            } else {
                dropTarget.element.parentNode.insertBefore(dragState.dropIndicator, dropTarget.element.nextSibling);
            }
        }
    }
}

// 获取放置目标位置
function getDropTarget(e) {
    // 获取所有图标栏(左右两侧)
    const leftIconBar = document.querySelector('.left-icon-bar');
    const rightIconBar = document.querySelector('.right-icon-bar');
    const iconBars = [leftIconBar, rightIconBar];

    let closestTarget = null;
    let closestDistance = Infinity;

    iconBars.forEach(iconBar => {
        const barRect = iconBar.getBoundingClientRect();

        // 扩大横轴的有效范围: 左右各扩展 100px
        const expandedLeft = barRect.left - 100;
        const expandedRight = barRect.right + 100;

        // 检查鼠标横轴是否在扩展后的图标栏范围内
        if (e.clientX >= expandedLeft && e.clientX <= expandedRight) {
            // 获取该图标栏的上下两个分组
            const topGroup = iconBar.querySelector('.icon-group-top');
            const bottomGroup = iconBar.querySelector('.icon-group-bottom');
            const groups = [
                { element: topGroup, position: 'top' },
                { element: bottomGroup, position: 'bottom' }
            ];

            groups.forEach(({ element: group }) => {
                if (!group) return;

                const groupRect = group.getBoundingClientRect();
                const buttons = group.querySelectorAll('.icon-btn:not(.dragging)');

                if (buttons.length === 0) {
                    // 空分组处理
                    // 计算鼠标到分组区域的距离
                    let distanceToGroup;

                    if (e.clientY < groupRect.top) {
                        distanceToGroup = groupRect.top - e.clientY;
                    } else if (e.clientY > groupRect.bottom) {
                        distanceToGroup = e.clientY - groupRect.bottom;
                    } else {
                        distanceToGroup = 0; // 鼠标在分组内
                    }

                    // 为空分组增加更大的容错范围(纵轴方向150px)
                    if (distanceToGroup <= 150) {
                        if (distanceToGroup < closestDistance) {
                            closestDistance = distanceToGroup;
                            closestTarget = {
                                element: group,
                                before: false,
                                isEmpty: true
                            };
                        }
                    }
                } else {
                    // 有按钮的分组
                    buttons.forEach(btn => {
                        const btnRect = btn.getBoundingClientRect();
                        const btnCenterY = btnRect.top + btnRect.height / 2;
                        const distance = Math.abs(e.clientY - btnCenterY);

                        if (distance < closestDistance) {
                            closestDistance = distance;
                            closestTarget = {
                                element: btn,
                                before: e.clientY < btnCenterY,
                                isEmpty: false
                            };
                        }
                    });
                }
            });
        }
    });

    return closestTarget;
}

// 完成拖动
function finishDragging(e) {
    const dropTarget = getDropTarget(e);

    if (dropTarget) {
        // 执行图标重新排列
        moveButton(dragState.draggedButton, dropTarget);
    }

    // 清理拖动状态
    if (dragState.dragGhost) {
        dragState.dragGhost.remove();
        dragState.dragGhost = null;
    }

    if (dragState.dropIndicator && dragState.dropIndicator.parentNode) {
        dragState.dropIndicator.remove();
    }
    dragState.dropIndicator = null;

    if (dragState.draggedButton) {
        dragState.draggedButton.classList.remove('dragging');
    }

    dragState.isDragging = false;
    dragState.draggedButton = null;
}

// 移动按钮到新位置
function moveButton(button, dropTarget) {
    // 获取按钮原来的信息
    const oldSide = button.dataset.side;
    const oldPosition = button.dataset.position;
    const wasActive = button.classList.contains('active');
    const panel = button.dataset.panel;

    // 确定新的side和position
    let newSide, newPosition;

    if (dropTarget.isEmpty) {
        // 空分组
        const group = dropTarget.element;
        const iconBar = group.closest('.left-icon-bar, .right-icon-bar');
        newSide = iconBar.classList.contains('left-icon-bar') ? 'left' : 'right';
        newPosition = group.classList.contains('icon-group-top') ? 'top' : 'bottom';

        // 移动按钮到空分组
        group.appendChild(button);
    } else {
        // 有按钮的分组
        const targetButton = dropTarget.element;
        const group = targetButton.closest('.icon-group-top, .icon-group-bottom');
        const iconBar = group.closest('.left-icon-bar, .right-icon-bar');
        newSide = iconBar.classList.contains('left-icon-bar') ? 'left' : 'right';
        newPosition = group.classList.contains('icon-group-top') ? 'top' : 'bottom';

        // 移动按钮
        if (dropTarget.before) {
            group.insertBefore(button, targetButton);
        } else {
            group.insertBefore(button, targetButton.nextSibling);
        }
    }

    // 更新按钮的数据属性
    button.dataset.side = newSide;
    button.dataset.position = newPosition;

    // 处理激活状态变化
    if (wasActive) {
        // 清除旧位置的激活状态
        activeButtons[oldSide][oldPosition] = null;

        // 检查新位置是否已有激活按钮
        if (activeButtons[newSide][newPosition]) {
            activeButtons[newSide][newPosition].classList.remove('active');
        }

        // 设置新位置的激活状态
        activeButtons[newSide][newPosition] = button;

        // 更新内容
        updateSidebarContent(newSide, newPosition, panel);

        // 更新两侧的侧边栏状态
        updateSidebarState(oldSide);
        updateSidebarState(newSide);
    }

    // 保存按钮布局
    saveButtonLayout();
}

// 保存按钮布局到localStorage
function saveButtonLayout() {
    const layout = {
        left: { top: [], bottom: [] },
        right: { top: [], bottom: [] }
    };

    // 收集所有按钮的布局信息
    document.querySelectorAll('.icon-btn').forEach(btn => {
        const side = btn.dataset.side;
        const position = btn.dataset.position;
        const panel = btn.dataset.panel;

        layout[side][position].push(panel);
    });

    try {
        localStorage.setItem('buttonLayout_testPage', JSON.stringify(layout));
    } catch (e) {
        console.error('保存按钮布局失败', e);
    }
}

// 从localStorage恢复按钮布局
function restoreButtonLayout() {
    try {
        const savedLayout = localStorage.getItem('buttonLayout_testPage');
        if (!savedLayout) return;

        const layout = JSON.parse(savedLayout);

        // 重新组织按钮
        Object.keys(layout).forEach(side => {
            Object.keys(layout[side]).forEach(position => {
                const panels = layout[side][position];
                const group = document.querySelector(`.${side}-icon-bar .icon-group-${position}`);

                panels.forEach(panel => {
                    const button = document.querySelector(`.icon-btn[data-panel="${panel}"]`);
                    if (button && group) {
                        button.dataset.side = side;
                        button.dataset.position = position;
                        group.appendChild(button);
                    }
                });
            });
        });
    } catch (e) {
        console.error('恢复按钮布局失败', e);
    }
}

// 页面加载时初始化侧边栏
document.addEventListener('DOMContentLoaded', function () {
    restoreButtonLayout(); // 先恢复布局
    initializeSidebar();
});

// =====
// 全局变量定义区
// =====

// --- 原始数据相关 ---
let filteredMatrix = null;
let currentXData = [];
let currentYData = [];
let currentXLabels = [];
let currentYLabels = [];
let xAvailableFields = [];
let yAvailableFields = [];
let availableCollections = [];

// --- 多图管理相关 ---
let allSimilarityResults = []; // 存储所有相似度矩阵结果，每个元素包含 visualConfig

// --- 全局默认配色（非可视化控制值的一部分，是全局UI设置） ---
let currentColorScheme = 'viridis';
let differenceMatrices = {}; // 存储差值矩阵的缓存 key格式: "idx1-idx2"
let currentMatrixIndex = 0; // 当前显示的矩阵索引

// =====
// 新架构：全局UI状态管理（与图表配置完全分离）
// =====

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

    // 临时筛选器：非独占模式下的UI控件值，与已启用筛选器做AND运算
    // 进入独占模式或差值模式时重置
    temporaryFilter: {
        enabled: false,  // 是否启用临时筛选器（当用户调整UI控件时自动启用）
        similarityRange: { min: 0, max: 1 },
        topK: { value: 0, axis: 'x' }
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

// =====
// 可视化配置管理
// =====

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


// =====
// 常量配置
// =====

// 颜色方案配置
const colorSchemes = {
    viridis: 'Viridis',
    plasma: 'Plasma',
    cividis: 'Cividis',
    hot: 'Hot',
    YlGnBu: 'YlGnBu',

};

// =====
// UI辅助函数
// =====

// 消息显示函数
/**
 * 显示右上角临时消息
 * @param {string} type - 消息类型: 'error', 'success', 'warning', 'info'
 * @param {string} message - 消息内容
 * @param {number} duration - 显示时长(毫秒),默认根据类型自动设置
 */
function showMessage(type, message, duration = 0) {
    // 根据类型设置默认显示时长
    if (duration === 0) {
        const defaultDurations = {
            'error': 5000,
            'success': 3000,
            'warning': 4000,
            'info': 3000
        };
        duration = defaultDurations[type] || 3000;
    }

    // 创建消息元素
    const messageEl = document.createElement('div');
    messageEl.className = `temp-message ${type}`;
    messageEl.textContent = message;

    // 添加到页面
    document.body.appendChild(messageEl);

    // 自动移除
    setTimeout(() => {
        messageEl.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (document.body.contains(messageEl)) {
                document.body.removeChild(messageEl);
            }
        }, 300);
    }, duration);
}

function showError(message, duration = 5000) { showMessage('error', message, duration); }
function showSuccess(message, duration = 3000) { showMessage('success', message, duration); }
function showWarning(message, duration = 4000) { showMessage('warning', message, duration); }
function showInfo(message, duration = 3000) { showMessage('info', message, duration); }

/**
 * 截断collection名称
 * @param {string} name - collection名称
 * @param {number} maxLength - 最大长度
 * @returns {string} 截断后的名称
 */
function truncateCollectionName(name, maxLength = 20) {
    if (!name) return '';
    if (name.length <= maxLength) return name;
    return name.substring(0, maxLength - 3) + '...';
}

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
        if (field) field.classList.add('required-empty');
        if (errorDiv) errorDiv.style.display = 'block';
    } else {
        if (field) field.classList.remove('required-empty');
        if (errorDiv) errorDiv.style.display = 'none';
    }
}

// 清除所有验证错误状态
function clearValidationErrors() {
    const xError = document.getElementById('xCollectionError');
    const yError = document.getElementById('yCollectionError');

    if (xError) xError.style.display = 'none';
    if (yError) yError.style.display = 'none';
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

/**
 * 文本自动换行函数 - 用于热力图悬浮提示
 * @param {string} text - 需要换行的文本
 * @param {number} maxCharsPerLine - 每行最大字符数（默认50）
 * @returns {string} - 换行后的文本
 */
function wrapTextForTooltip(text, maxCharsPerLine = 50) {
    if (!text || typeof text !== 'string') {
        return 'N/A';
    }

    // 如果文本长度小于等于最大字符数，直接返回
    if (text.length <= maxCharsPerLine) {
        return text;
    }

    const lines = [];
    let currentPos = 0;

    while (currentPos < text.length) {
        // 如果剩余文本小于最大字符数，直接添加
        if (currentPos + maxCharsPerLine >= text.length) {
            lines.push(text.substring(currentPos));
            break;
        }

        // 尝试在最大字符数附近找到合适的断点（空格、标点符号等）
        let breakPos = currentPos + maxCharsPerLine;
        const searchStart = Math.max(currentPos, breakPos - 10); // 向前搜索10个字符

        // 查找最近的空格或标点符号作为断点
        const breakChars = [' ', '，', '。', '、', '；', '：', '！', '？', ',', '.', ';', ':', '!', '?'];
        let foundBreak = false;

        for (let i = breakPos; i >= searchStart; i--) {
            if (breakChars.includes(text[i])) {
                breakPos = i + 1; // 在标点符号后断开
                foundBreak = true;
                break;
            }
        }

        // 如果没找到合适的断点，就在最大字符数处强制断开
        if (!foundBreak) {
            breakPos = currentPos + maxCharsPerLine;
        }

        lines.push(text.substring(currentPos, breakPos));
        currentPos = breakPos;
    }

    // 使用 HTML <br> 标签连接各行
    return lines.join('<br>');
}

// =====
// API调用与数据加载
// =====

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
    removeBtn.onclick = function () {
        removeCollectionSelect(this);
    };

    newRow.appendChild(newSelect);
    newRow.appendChild(removeBtn);
    container.appendChild(newRow);

    // 清除验证错误
    const fieldId = axis === 'x' ? 'xCollection' : 'yCollection';
    showFieldValidationError(fieldId, false);
}

// 删除collection选择框
function removeCollectionSelect(button) {
    const row = button.closest('.collection-select-row');
    row.remove();
}


// =====
// 布尔矩阵计算函数（新架构核心）
// =====

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
 * 获取矩阵的尺寸信息
 * @param {number} index - 图表索引
 * @returns {Object|null} - 返回 {rows, cols} 或 null
 */
function getMatrixSize(index) {
    if (index < 0 || index >= allSimilarityResults.length) {
        return null;
    }
    const matrix = allSimilarityResults[index].matrix;
    return {
        rows: matrix.length,
        cols: matrix.length > 0 ? matrix[0].length : 0
    };
}

/**
 * 检查指定图表的矩阵大小是否与当前已启用的图表一致
 * @param {number} newIndex - 要检查的图表索引
 * @param {string} buttonType - 按钮类型: 'applyData' 或 'applyFilter'
 * @returns {Object} - {isValid: boolean, message: string, conflictIndices: Array}
 */
function checkMatrixSizeConsistency(newIndex, buttonType) {
    const newSize = getMatrixSize(newIndex);
    if (!newSize) {
        return { isValid: false, message: '无效的图表索引', conflictIndices: [] };
    }

    let activeIndices = [];

    // 根据按钮类型确定需要检查的已启用图表
    if (buttonType === 'applyData') {
        // 检查"应用数据"按钮：需要与已启用的应用数据按钮一致
        if (globalUIState.dataSource.primaryIndex !== null) {
            activeIndices.push(globalUIState.dataSource.primaryIndex);
        }
        if (globalUIState.dataSource.subtractIndex !== null) {
            activeIndices.push(globalUIState.dataSource.subtractIndex);
        }
    } else if (buttonType === 'applyFilter') {
        // 检查"应用筛选器"按钮：需要与当前数据源一致（如果有数据源）
        // 同时也需要与其他已启用的筛选器一致
        if (globalUIState.dataSource.primaryIndex !== null) {
            activeIndices.push(globalUIState.dataSource.primaryIndex);
        }
        if (globalUIState.dataSource.subtractIndex !== null) {
            activeIndices.push(globalUIState.dataSource.subtractIndex);
        }
        // 添加其他已启用的筛选器
        globalUIState.filters.activeFilterIndices.forEach(idx => {
            if (!activeIndices.includes(idx)) {
                activeIndices.push(idx);
            }
        });
    }

    // 如果没有已启用的图表，直接允许
    if (activeIndices.length === 0) {
        return { isValid: true, message: '', conflictIndices: [] };
    }

    // 检查是否与所有已启用的图表尺寸一致
    const conflictIndices = [];
    for (const activeIdx of activeIndices) {
        const activeSize = getMatrixSize(activeIdx);
        if (activeSize && (activeSize.rows !== newSize.rows || activeSize.cols !== newSize.cols)) {
            conflictIndices.push(activeIdx);
        }
    }

    if (conflictIndices.length > 0) {
        const conflictList = conflictIndices.map(idx => `图表${idx + 1}`).join('、');
        return {
            isValid: false,
            message: `矩阵大小不一致 (当前图: ${newSize.rows}×${newSize.cols}，已启用的${conflictList}有不同尺寸)，请先松开原按钮`,
            conflictIndices: conflictIndices
        };
    }

    return { isValid: true, message: '', conflictIndices: [] };
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

/**
 * 计算临时筛选器的布尔遮罩
 * 使用当前显示的矩阵数据和临时筛选器配置
 * @returns {Array<Array<boolean>>|null} - 临时筛选器的布尔遮罩
 */
function computeTemporaryFilterMask() {
    if (!globalUIState.temporaryFilter.enabled) {
        return null;
    }

    const currentMatrix = globalUIState.dataSource.currentMatrix;
    if (!currentMatrix) {
        return null;
    }

    const tempFilter = globalUIState.temporaryFilter;

    console.log(`[临时筛选器] 计算遮罩 - 阈值: ${tempFilter.similarityRange.min.toFixed(2)}-${tempFilter.similarityRange.max.toFixed(2)}, Top-K: ${tempFilter.topK.value} (${tempFilter.topK.axis}轴)`);

    // 1. 计算阈值遮罩
    const thresholdMask = computeThresholdMask(
        currentMatrix,
        tempFilter.similarityRange.min,
        tempFilter.similarityRange.max
    );

    // 2. 计算Top-K遮罩
    const topKMask = computeTopKMask(
        currentMatrix,
        tempFilter.topK.value,
        tempFilter.topK.axis
    );

    // 3. AND合并
    return combineWithAND(thresholdMask, topKMask);
}

/**
 * 重置临时筛选器
 * @param {boolean} isInDifferenceMode - 是否为差值模式
 */
function resetTemporaryFilter(isInDifferenceMode = false) {
    const range = isInDifferenceMode ? { min: -1, max: 1 } : { min: 0, max: 1 };

    globalUIState.temporaryFilter.enabled = false;
    globalUIState.temporaryFilter.similarityRange = { ...range };
    globalUIState.temporaryFilter.topK = { value: 0, axis: 'x' };

    console.log(`[临时筛选器] 已重置 - 差值模式: ${isInDifferenceMode}, 范围: ${range.min} - ${range.max}`);
}

/**
 * 将临时筛选器的值应用到UI控件
 */
function applyTemporaryFilterToUI() {
    const tempFilter = globalUIState.temporaryFilter;

    // 注意：这里不更新UI控件的值，因为UI控件本身就是用户交互的源头
    // 这个函数主要用于重置后，确保UI状态同步
    // 实际的UI更新在applyFilterStateToUI中完成
    console.log(`[临时筛选器] UI已同步 - enabled: ${tempFilter.enabled}`);
}

// =====
// 数据处理与标签生成
// =====

// 生成唯一标签函数，使用零宽字符和不间断空格确保唯一性
function generateUniqueLabels(data, field) {
    // 使用 Map 来跟踪每个值的出现次数
    const valueCountMap = new Map();

    return data.map((item) => {
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

// =====
// 可视化控制与筛选
// =====

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
    } else {
        // *** 非独占模式：更新临时筛选器 ***
        globalUIState.temporaryFilter.enabled = true;
        globalUIState.temporaryFilter.topK.axis = axis;
        console.log(`[临时筛选器] 更新Top-K轴: ${axis}`);
    }

    // 更新按钮状态
    document.getElementById('xAxisBtn').classList.toggle('active', axis === 'x');
    document.getElementById('yAxisBtn').classList.toggle('active', axis === 'y');

    // 更新Top-K滑块的最大值
    updateTopkSliderMax();

    // 更新热力图
    if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
        updateHeatmap();
        // 更新统计信息
        if (globalUIState.dataSource.subtractIndex !== null) {
            showDifferenceStatistics(false);
        } else {
            showStatistics(false);
        }
    }
}

// =====
// 相似度计算与矩阵管理
// =====

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

    const xMaxItemsEl = document.getElementById('xMaxItems');
    const yMaxItemsEl = document.getElementById('yMaxItems');

    if (!xMaxItemsEl || !yMaxItemsEl) {
        showError('无法找到配置元素，请确保数据源配置面板已打开');
        return;
    }

    const xMaxItems = parseInt(xMaxItemsEl.value) || 30;
    const yMaxItems = parseInt(yMaxItemsEl.value) || 30;

    // 先清除之前的验证错误
    clearValidationErrors();

    // 检查必要项是否填写
    if (xCollections.length === 0) {
        const xErrorEl = document.getElementById('xCollectionError');
        if (xErrorEl) xErrorEl.style.display = 'block';
        showError('请至少选择一个横坐标 Collection');
        return;
    }
    if (yCollections.length === 0) {
        const yErrorEl = document.getElementById('yCollectionError');
        if (yErrorEl) yErrorEl.style.display = 'block';
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
                    // 单个矩阵计算失败,显示警告并继续下一组
                    console.warn(`[相似度计算] ${xCollection} vs ${yCollection} 失败:`, error);
                    showWarning(`跳过 ${xCollection} vs ${yCollection}: ${error.message}`, 4000);
                    // 不添加到结果中,继续下一个
                }
            }
        }

        if (allSimilarityResults.length === 0) {
            throw new Error('所有相似度计算都失败了');
        }

        // *** 新架构：初始化按钮状态数组 ***
        initializeButtonStates();

        // 重置全局UI状态
        globalUIState.dataSource.primaryIndex = null;
        globalUIState.dataSource.subtractIndex = null;
        globalUIState.dataSource.currentMatrix = null;
        globalUIState.filters.activeFilterIndices = [];
        globalUIState.exclusiveMode.active = false;
        globalUIState.exclusiveMode.editingIndex = null;

        // *** 新架构：更新图表列表UI ***
        updateMatrixListUI();
        updateExportMatrixSelector();

        // 显示图表选择控制区域 (添加空值检查)
        const matrixSelectorControl = document.getElementById('matrixSelectorControl');
        if (matrixSelectorControl) {
            matrixSelectorControl.style.display = 'block';
        }

        // *** 新特性:首次计算相似度时自动启用第一张图的独占模式 ***
        if (allSimilarityResults.length > 0) {
            console.log('[自动独占] 首次计算相似度，自动启用图0的独占模式');
            await enterExclusiveMode(0);
            updateMatrixListUI(); // 再次更新UI以反映独占模式状态
        }

        showSuccess(`成功计算 ${allSimilarityResults.length} 个相似度矩阵! 已自动启用第一张图的编辑模式`);

    } catch (error) {
        showError('计算相似度失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// =====
// 可视化更新与渲染
// =====

// 更新Top-K滑块的最大值
function updateTopkSliderMax() {
    if (!filteredMatrix) return;

    const maxX = filteredMatrix[0] ? filteredMatrix[0].length : 0;
    const maxY = filteredMatrix.length;

    // 根据当前选择的轴确定最大值
    const maxTopk = currentTopkAxis === 'x' ? maxX : maxY;

    const topkSlider = document.getElementById('topkSlider');
    if (!topkSlider) return;

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
        // 手动触发 input 事件，以确保状态同步和缓存失效
        topkSlider.dispatchEvent(new Event('input'));
    }

    // 确保按钮状态正确更新
    updateTopkButtons();
}

// 调整最小相似度值
function adjustMinSimilarity(delta) {
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const currentValue = parseFloat(minSlider.value);
    const minValue = parseFloat(minSlider.min);
    const maxValue = parseFloat(maxSlider.value);
    const newValue = Math.max(minValue, Math.min(maxValue, currentValue + delta));

    // 保留两位小数避免浮点精度问题
    const roundedValue = Math.round(newValue * 100) / 100;

    if (roundedValue !== currentValue) {
        minSlider.value = roundedValue;
        // 手动触发 input 事件，以确保状态同步和缓存失效
        minSlider.dispatchEvent(new Event('input'));
    }
}

// 调整最大相似度值
function adjustMaxSimilarity(delta) {
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const currentValue = parseFloat(maxSlider.value);
    const minValue = parseFloat(minSlider.value);
    const maxValue = parseFloat(maxSlider.max);
    const newValue = Math.max(minValue, Math.min(maxValue, currentValue + delta));

    // 保留两位小数避免浮点精度问题
    const roundedValue = Math.round(newValue * 100) / 100;

    if (roundedValue !== currentValue) {
        maxSlider.value = roundedValue;
        // 手动触发 input 事件，以确保状态同步和缓存失效
        maxSlider.dispatchEvent(new Event('input'));
    }
}

/**
 * 计算当前应用的最终布尔矩阵遮罩
 * @param {boolean} needTopKAxis - 是否需要返回topKAxis信息(用于统计面板显示缺失匹配)
 * @returns {Object|Array} - 如果needTopKAxis=true返回{finalMask, topKAxis}, 否则只返回finalMask
 */
function computeCurrentFinalMask(needTopKAxis = false) {
    if (!filteredMatrix) {
        return needTopKAxis ? { finalMask: null, topKAxis: 'none' } : null;
    }

    let finalMask = null;
    let topKAxis = 'none';

    // 情况1: 独占模式
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        const editingIndex = globalUIState.exclusiveMode.editingIndex;
        const config = allSimilarityResults[editingIndex].visualConfig;
        finalMask = config.cachedMasks?.finalMask || computeFinalMaskForMatrix(editingIndex);

        if (needTopKAxis) {
            topKAxis = config.filters.topK.value > 0 ? config.filters.topK.axis : 'none';
        }
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

        // 应用临时筛选器 (AND运算)
        const tempFilterMask = computeTemporaryFilterMask();
        if (tempFilterMask) {
            finalMask = finalMask ? combineWithAND(finalMask, tempFilterMask) : tempFilterMask;

            if (needTopKAxis) {
                topKAxis = globalUIState.temporaryFilter.topK.value > 0
                    ? globalUIState.temporaryFilter.topK.axis
                    : 'none';
            }
        }
    }
    // 情况3: 默认模式
    else {
        const tempFilterMask = computeTemporaryFilterMask();
        if (tempFilterMask) {
            finalMask = tempFilterMask;

            if (needTopKAxis) {
                topKAxis = globalUIState.temporaryFilter.topK.value > 0
                    ? globalUIState.temporaryFilter.topK.axis
                    : 'none';
            }
        } else {
            const minSim = parseFloat(document.getElementById('minSimilaritySlider')?.value || 0);
            const maxSim = parseFloat(document.getElementById('maxSimilaritySlider')?.value || 1);
            const topK = parseInt(document.getElementById('topkSlider')?.value || 0);
            const axis = currentTopkAxis || 'x';

            const thresholdMask = computeThresholdMask(filteredMatrix, minSim, maxSim);
            const topKMask = computeTopKMask(filteredMatrix, topK, axis);
            finalMask = combineWithAND(thresholdMask, topKMask);

            if (needTopKAxis) {
                topKAxis = topK > 0 ? axis : 'none';
            }
        }
    }

    return needTopKAxis ? { finalMask, topKAxis } : finalMask;
}

// 计算当前显示的对比数 - 新架构：使用布尔矩阵
function getCurrentDisplayCount() {
    const finalMask = computeCurrentFinalMask(false);

    if (!finalMask) return 0;

    // 统计true的数量
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

    // =====
    // 新架构核心逻辑 - 使用公共函数计算最终遮罩
    // =====

    // 添加日志以保持调试能力
    if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
        console.log(`[updateHeatmap] 独占模式 - 使用图 ${globalUIState.exclusiveMode.editingIndex} 的遮罩`);
    } else if (globalUIState.filters.activeFilterIndices.length > 0) {
        console.log(`[updateHeatmap] 应用筛选器模式 - 合并 ${globalUIState.filters.activeFilterIndices.length} 个图的遮罩`);
    } else {
        console.log('[updateHeatmap] 默认模式 - 使用临时筛选器');
    }

    // 使用公共函数计算最终遮罩
    const finalMask = computeCurrentFinalMask(false);

    // 应用遮罩到原始数据
    let displayMatrix = filteredMatrix;
    if (finalMask) {
        displayMatrix = applyMask(filteredMatrix, finalMask);
    }

    let displayXLabels = [...currentXLabels];
    let displayYLabels = [...currentYLabels];
    // 更新热力图数据，但保持当前的缩放状态
    updateHeatmapData(displayMatrix, displayXLabels, displayYLabels, currentLayout);

    // 更新统计信息中的当前显示对比数
    updateCurrentDisplayStat();
}

// 更新热力图数据但保持缩放状态
function updateHeatmapData(matrix, xLabels, yLabels, preserveLayout = null) {
    // 获取当前模式的相似度范围
    const range = getCurrentSimilarityRange();

    // 准备 customdata：为每个格点存储完整的显示文本（支持换行）
    const xDisplayField = document.getElementById('xDisplayField').value;
    const yDisplayField = document.getElementById('yDisplayField').value;

    const customdata = matrix.map((row, yIdx) =>
        row.map((val, xIdx) => {
            // 获取原始数据项
            const xItem = currentXData[xIdx];
            const yItem = currentYData[yIdx];

            // 获取完整的字段值
            let xFullText = '';
            let yFullText = '';

            if (xItem) {
                if (xDisplayField === 'order_id') {
                    xFullText = `ID-${xItem[xDisplayField]}`;
                } else {
                    xFullText = String(xItem[xDisplayField] || 'N/A');
                }
            }

            if (yItem) {
                if (yDisplayField === 'order_id') {
                    yFullText = `ID-${yItem[yDisplayField]}`;
                } else {
                    yFullText = String(yItem[yDisplayField] || 'N/A');
                }
            }

            // 应用自动换行
            return {
                xText: wrapTextForTooltip(xFullText, 50),
                yText: wrapTextForTooltip(yFullText, 50)
            };
        })
    );
    isInDifferenceMode() ? '差值' : '相似度';
    isInDifferenceMode() ? 0.2 : 0.1;

    if (preserveLayout) {
        document.getElementById('heatmap');
        // 使用 Plotly.restyle 只更新数据，不影响布局
        Plotly.restyle('heatmap', {
            z: [matrix],
            x: [xLabels],
            y: [yLabels],
            customdata: [customdata],  // 同时更新 customdata
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

// 更新当前显示对比数统计
function updateCurrentDisplayStat() {
    const currentDisplayElement = document.getElementById('currentDisplayCount');
    if (currentDisplayElement) {
        const currentCount = getCurrentDisplayCount();

        // 获取总对比数
        let totalCount = 0;
        if (filteredMatrix && filteredMatrix.length > 0) {
            totalCount = filteredMatrix.length * filteredMatrix[0].length;
        } else if (globalUIState.dataSource.currentMatrix && globalUIState.dataSource.currentMatrix.length > 0) {
            totalCount = globalUIState.dataSource.currentMatrix.length * globalUIState.dataSource.currentMatrix[0].length;
        }

        // 统一格式: 当前/总数
        currentDisplayElement.textContent = `${currentCount} / ${totalCount}`;
    }
}


// 获取当前模式下的相似度范围
function getCurrentSimilarityRange() {
    return isInDifferenceMode() ? { min: -1, max: 1 } : { min: 0, max: 1 };
}


// 创建热力图
function createHeatmap(matrix = filteredMatrix, xLabels = currentXLabels, yLabels = currentYLabels) {
    if (!matrix) {
        showError('没有相似度数据');
        return;
    }

    // 获取当前模式的相似度范围
    const range = getCurrentSimilarityRange();

    // 准备 customdata：为每个格点存储完整的显示文本（支持换行）
    const xDisplayField = document.getElementById('xDisplayField').value;
    const yDisplayField = document.getElementById('yDisplayField').value;

    const customdata = matrix.map((row, yIdx) =>
        row.map((val, xIdx) => {
            // 获取原始数据项
            const xItem = currentXData[xIdx];
            const yItem = currentYData[yIdx];

            // 获取完整的字段值
            let xFullText = '';
            let yFullText = '';

            if (xItem) {
                if (xDisplayField === 'order_id') {
                    xFullText = `ID-${xItem[xDisplayField]}`;
                } else {
                    xFullText = String(xItem[xDisplayField] || 'N/A');
                }
            }

            if (yItem) {
                if (yDisplayField === 'order_id') {
                    yFullText = `ID-${yItem[yDisplayField]}`;
                } else {
                    yFullText = String(yItem[yDisplayField] || 'N/A');
                }
            }

            // 应用自动换行
            return {
                xText: wrapTextForTooltip(xFullText, 50),
                yText: wrapTextForTooltip(yFullText, 50)
            };
        })
    );

    const trace = {
        z: matrix,
        x: xLabels,
        y: yLabels,
        customdata: customdata,  // 添加 customdata
        type: 'heatmap',
        colorscale: colorSchemes[currentColorScheme],
        hoverongaps: false,
        // 修改 hovertemplate 使用 customdata 中的换行文本
        hovertemplate: '<b>Y: %{customdata.yText}</b><br>' +
            '<b>X: %{customdata.xText}</b><br>' +
            '<b>相似度: %{z:.4f}</b>' +
            '<extra></extra>',
        colorbar: {
            title: isInDifferenceMode() ? '差值' : '相似度',
            titleside: 'right',
            tickmode: 'linear',
            tick0: range.min,
            dtick: isInDifferenceMode() ? 0.2 : 0.1
        },
        showscale: true,
        zmin: range.min,  // 固定最小值
        zmax: range.max   // 固定最大值
    };

    // 获取容器实际尺寸
    const container = document.querySelector('.heatmap-container');
    const containerRect = container.getBoundingClientRect();

    // 计算可用空间（减去padding和controls的高度）
    const availableWidth = containerRect.width;
    const availableHeight = containerRect.height;

    // 动态生成坐标轴标题
    const matrixData = allSimilarityResults[currentMatrixIndex];
    const xCollectionName = matrixData ? matrixData.xCollection : '';
    const yCollectionName = matrixData ? matrixData.yCollection : '';

    // 注意：xDisplayField 和 yDisplayField 已在函数开头声明，这里直接使用
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
        hoverlabel: {
            bgcolor: 'rgba(255, 255, 255, 0.2)',  // 白色背景,透明度90%
            bordercolor: '#404040',
            font: {
                size: 12,
                color: '#404040'
            }
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

// 调整热力图大小以适应容器变化
function resizeHeatmap() {
    const heatmapDiv = document.getElementById('heatmap');
    if (!heatmapDiv || !heatmapDiv.data) {
        return;
    }

    // 使用requestAnimationFrame确保在下一次重绘前执行
    requestAnimationFrame(() => {
        // 获取heatmap-container的实际尺寸
        const container = document.querySelector('.heatmap-container');
        if (!container) return;

        const containerRect = container.getBoundingClientRect();

        // 计算可用空间（减去padding和controls的高度）
        const availableWidth = containerRect.width;
        document.querySelector('.heatmap-controls');
        const availableHeight = containerRect.height;

        console.log(`[热力图Resize] 容器尺寸: ${containerRect.width}x${containerRect.height}, 可用尺寸: ${availableWidth}x${availableHeight}`);

        // 使用update强制更新图表尺寸(同时更新data和layout,解决缩小时不生效的问题)
        Plotly.update(heatmapDiv, {}, {
            width: availableWidth,
            height: availableHeight
        }).catch(err => {
            console.warn('[热力图Resize] update失败', err);
        });
    });
}

/**
 * 计算非差值模式下的统计信息
 * @param {Array<Array<boolean>>} booleanMask - 当前作用于画面的布尔矩阵
 * @param {string} topKAxis - Top-K轴向 ('x' 或 'y' 或 'none')
 * @returns {Object} - 统计信息对象
 */
function calculateNormalModeStatistics(booleanMask, topKAxis = 'none') {
    if (!booleanMask || booleanMask.length === 0) {
        return {
            totalCount: 0,
            currentDisplayCount: 0,
            diagonalTrueCount: 0,
            diagonalTotalCount: 0,
            missingMatchCount: 0
        };
    }

    const rows = booleanMask.length;
    const cols = booleanMask[0].length;

    // 1. 统计布尔矩阵中true的总数
    let totalTrueCount = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (booleanMask[i][j]) {
                totalTrueCount++;
            }
        }
    }

    // 2. 统计斜对角线上的true数量 (左上到右下的对角线)
    let diagonalTrueCount = 0;
    const diagonalTotalCount = Math.min(rows, cols); // 斜对角线总格数
    for (let i = 0; i < diagonalTotalCount; i++) {
        if (booleanMask[i][i]) {
            diagonalTrueCount++;
        }
    }

    // 3. 统计缺失匹配数量
    let missingMatchCount = 0;
    if (topKAxis === 'x') {
        // 横轴Top-K: 统计有几行全是false
        for (let i = 0; i < rows; i++) {
            let hasTrue = false;
            for (let j = 0; j < cols; j++) {
                if (booleanMask[i][j]) {
                    hasTrue = true;
                    break;
                }
            }
            if (!hasTrue) {
                missingMatchCount++;
            }
        }
    } else if (topKAxis === 'y') {
        // 纵轴Top-K: 统计有几列全是false
        for (let j = 0; j < cols; j++) {
            let hasTrue = false;
            for (let i = 0; i < rows; i++) {
                if (booleanMask[i][j]) {
                    hasTrue = true;
                    break;
                }
            }
            if (!hasTrue) {
                missingMatchCount++;
            }
        }
    }

    const totalCount = rows * cols;

    return {
        totalCount: totalCount,
        currentDisplayCount: totalTrueCount,
        diagonalTrueCount: diagonalTrueCount,
        diagonalTotalCount: diagonalTotalCount,
        missingMatchCount: missingMatchCount
    };
}

/**
 * 计算差值模式下的统计信息
 * @param {Array<Array<boolean>>} groundTruthMask - 被减数图的布尔矩阵(ground truth)
 * @param {Array<Array<boolean>>} currentMask - 当前作用于画面的布尔矩阵
 * @returns {Object} - 统计信息对象
 */
function calculateDifferenceModeStatistics(groundTruthMask, currentMask) {
    if (!groundTruthMask || !currentMask ||
        groundTruthMask.length === 0 || currentMask.length === 0) {
        return {
            truePositive: 0,
            trueNegative: 0,
            falsePositive: 0,
            falseNegative: 0,
            contextRecall: 0,
            contextPrecision: 0
        };
    }

    const rows = groundTruthMask.length;
    const cols = groundTruthMask[0].length;

    let truePositive = 0;   // ground_truth为1, 当前数据为1
    let trueNegative = 0;   // ground_truth为0, 当前数据为0
    let falsePositive = 0;  // ground_truth为0, 当前数据为1
    let falseNegative = 0;  // ground_truth为1, 当前数据为0

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const gtValue = groundTruthMask[i][j];
            const currentValue = currentMask[i][j];

            if (gtValue && currentValue) {
                truePositive++;
            } else if (!gtValue && !currentValue) {
                trueNegative++;
            } else if (!gtValue && currentValue) {
                falsePositive++;
            } else if (gtValue && !currentValue) {
                falseNegative++;
            }
        }
    }

    // 计算上下文召回率 = TP / (TP + FN)
    const contextRecall = (truePositive + falseNegative) > 0
        ? truePositive / (truePositive + falseNegative)
        : 0;

    // 计算上下文精度 = TP / (TP + FP)
    const contextPrecision = (truePositive + falsePositive) > 0
        ? truePositive / (truePositive + falsePositive)
        : 0;

    return {
        truePositive: truePositive,
        trueNegative: trueNegative,
        falsePositive: falsePositive,
        falseNegative: falseNegative,
        contextRecall: contextRecall,
        contextPrecision: contextPrecision
    };
}

// 显示统计信息
function showStatistics(activatePanel = true) {
    const statsGrid = document.getElementById('statsGrid');

    if (!statsGrid) {
        console.warn('statsGrid元素不存在，统计信息面板可能未加载');
        return;
    }

    // 使用公共函数获取当前应用的最终布尔矩阵和topKAxis
    const { finalMask, topKAxis } = computeCurrentFinalMask(true);

    if (!finalMask) {
        statsGrid.innerHTML = '<div style="padding: 10px; text-align: center; color: #999;">暂无统计数据</div>';
        return;
    }

    // 计算统计信息
    const stats = calculateNormalModeStatistics(finalMask, topKAxis);

    // 构建HTML
    let statsHTML = '';

    // 1. 当前显示对比数/总对比数
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-eyeglasses"></i></div>
            <div class="label">当前显示对比数</div>
            <div class="value" id="currentDisplayCount">${stats.currentDisplayCount} / ${stats.totalCount}</div>
        </div>
    `;

    // 2. 斜对角线上对比数/斜对角线总格数
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-shadows"></i></div>
            <div class="label">斜对角线对比数</div>
            <div class="value">${stats.diagonalTrueCount} / ${stats.diagonalTotalCount}</div>
        </div>
    `;

    // 3. 缺失匹配 (仅在激活Top-K时显示)
    if (topKAxis === 'x' || topKAxis === 'y') {
        const axisLabel = topKAxis === 'x' ? '横轴' : '纵轴';
        statsHTML += `
            <div class="stat-item">
                <div class="icon"><i class="bi bi-exclamation-triangle-fill"></i></div>
                <div class="label">缺失匹配(${axisLabel})</div>
                <div class="value">${stats.missingMatchCount}</div>
            </div>
        `;
    }

    statsGrid.innerHTML = statsHTML;

    if (activatePanel) {
        // 激活统计信息按钮（使用新架构）
        const statisticsBtn = document.querySelector('[data-side="right"][data-position="top"][data-panel="statistics"]');
        if (statisticsBtn && !statisticsBtn.classList.contains('active')) {
            // 取消同位置其他按钮的激活状态
            if (activeButtons.right.top && activeButtons.right.top !== statisticsBtn) {
                activeButtons.right.top.classList.remove('active');
            }
            statisticsBtn.classList.add('active');
            activeButtons.right.top = statisticsBtn;

            // 关键：调用 updateSidebarContent 来移动面板
            updateSidebarContent('right', 'top', 'statistics');
        }

        // 激活筛选器控制按钮（使用新架构）
        const filtersBtn = document.querySelector('[data-side="right"][data-position="bottom"][data-panel="filters"]');
        if (filtersBtn && !filtersBtn.classList.contains('active')) {
            // 取消同位置其他按钮的激活状态
            if (activeButtons.right.bottom && activeButtons.right.bottom !== filtersBtn) {
                activeButtons.right.bottom.classList.remove('active');
            }
            filtersBtn.classList.add('active');
            activeButtons.right.bottom = filtersBtn;

            // 关键：调用 updateSidebarContent 来移动面板
            updateSidebarContent('right', 'bottom', 'filters');
        }

        // 更新右侧边栏状态（打开侧边栏）
        updateSidebarState('right');
    }
}

// =====
// UI控件初始化
// =====

// 计算上色条的位置和宽度(考虑thumb宽度)
function calculateTrackPosition(minValue, maxValue, minSlider) {
    const sliderMin = parseFloat(minSlider.min);
    const sliderMax = parseFloat(minSlider.max);
    const sliderWidth = minSlider.offsetWidth;
    const thumbWidth = 16; // 滑块宽度
    const thumbRadius = thumbWidth / 2;

    // 如果元素还没有渲染(offsetWidth为0),返回默认值
    if (sliderWidth === 0) {
        console.warn('Slider width is 0, element not yet rendered');
        return {
            left: 0,
            width: 0
        };
    }

    // 计算值的百分比
    const percent1 = (minValue - sliderMin) / (sliderMax - sliderMin);
    const percent2 = (maxValue - sliderMin) / (sliderMax - sliderMin);

    // 计算thumb中心点的实际像素位置
    const availableWidth = sliderWidth - thumbWidth * 2;
    const leftPos = thumbRadius + availableWidth * percent1;
    const rightPos = thumbRadius + availableWidth * percent2;

    return {
        left: leftPos + 2,
        width: rightPos - leftPos
    };
}

// 初始化双滑块
function initRangeSlider() {
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');
    const track = document.getElementById('similarityTrack');

    // 如果任何必需元素不存在,退出初始化
    if (!minSlider || !maxSlider || !minInput || !maxInput || !track) {
        console.warn('Range slider elements not found, skipping initialization');
        return;
    }

    function updateTrack() {
        const min = parseFloat(minSlider.value);
        const max = parseFloat(maxSlider.value);

        // 使用辅助函数计算上色条位置
        const pos = calculateTrackPosition(min, max, minSlider);
        track.style.left = pos.left + 'px';
        track.style.width = pos.width + 'px';

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
        } else {
            // *** 非独占模式：更新临时筛选器 ***
            globalUIState.temporaryFilter.enabled = true;
            globalUIState.temporaryFilter.similarityRange.min = min;
            globalUIState.temporaryFilter.similarityRange.max = max;
            console.log(`[临时筛选器] 更新阈值范围: ${min.toFixed(2)} - ${max.toFixed(2)}`);
        }

        // 防抖: 延迟100ms更新热力图和统计信息
        clearTimeout(heatmapUpdateTimer);
        heatmapUpdateTimer = setTimeout(() => {
            if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
                updateHeatmap();

                // 统计信息也使用防抖,在热力图更新后执行
                clearTimeout(statsUpdateTimer);
                statsUpdateTimer = setTimeout(() => {
                    if (globalUIState.dataSource.subtractIndex !== null) {
                        showDifferenceStatistics(false);
                    } else {
                        showStatistics(false);
                    }
                }, 50);
            }
        }, DEBOUNCE_DELAY);
    }

    minSlider.addEventListener('input', function () {
        if (parseFloat(this.value) > parseFloat(maxSlider.value)) {
            this.value = maxSlider.value;
        }
        updateTrack();
    });

    maxSlider.addEventListener('input', function () {
        if (parseFloat(this.value) < parseFloat(minSlider.value)) {
            this.value = minSlider.value;
        }
        updateTrack();
    });

    minInput.addEventListener('change', function () {
        // 修复：使用 ?? 而非 ||，避免将 0 当作 falsy 值
        const inputValue = parseFloat(this.value);
        const defaultValue = parseFloat(minSlider.min);
        minSlider.value = Math.max(parseFloat(minSlider.min), Math.min(parseFloat(minSlider.max), isNaN(inputValue) ? defaultValue : inputValue));
        updateTrack();
    });

    maxInput.addEventListener('change', function () {
        // 修复：使用 ?? 而非 ||，避免将 0 当作 falsy 值
        const inputValue = parseFloat(this.value);
        const defaultValue = parseFloat(maxSlider.max);
        maxSlider.value = Math.max(parseFloat(minSlider.min), Math.min(parseFloat(maxSlider.max), isNaN(inputValue) ? defaultValue : inputValue));
        updateTrack();
    });

    // 初始更新
    updateTrack();

    // 延迟更新,确保元素完全渲染后重新计算
    setTimeout(() => {
        if (minSlider.offsetWidth > 0) {
            updateTrack();
        }
    }, 100);

    // 暴露updateTrack到全局,以便在侧边栏打开时调用
    window.updateSimilarityTrack = updateTrack;
}

// 初始化Top-K滑块
function initTopkSlider() {
    const topkSlider = document.getElementById('topkSlider');
    const topkValue = document.getElementById('topkValue');
    const topkStatus = document.getElementById('topkStatus');

    // 如果必需元素不存在,退出初始化
    if (!topkSlider || !topkValue || !topkStatus) {
        console.warn('Top-K slider elements not found, skipping initialization');
        return;
    }

    function updateTopkDisplayInternal() {
        const topkSlider = document.getElementById('topkSlider');
        const topkValueEl = document.getElementById('topkValue');
        const topkStatusEl = document.getElementById('topkStatus');

        if (!topkSlider || !topkValueEl || !topkStatusEl) return;

        const topkVal = parseInt(topkSlider.value);
        topkValueEl.textContent = topkVal;
        topkStatusEl.textContent = topkVal === 0 ? '显示全部' : `显示Top-${topkVal}`;

        // 更新全局筛选器状态
        globalUIState.filters.uiState.topK.value = topkVal;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.filters.topK.value = topkVal;

            // 标记布尔矩阵缓存失效
            if (config.cachedMasks) {
                config.cachedMasks.topKMask = null;
                config.cachedMasks.finalMask = null;
            }

            console.log(`[独占模式] 保存Top-K值: ${topkVal}，缓存已失效`);
        } else {
            // *** 非独占模式：更新临时筛选器 ***
            globalUIState.temporaryFilter.enabled = true;
            globalUIState.temporaryFilter.topK.value = topkVal;
            console.log(`[临时筛选器] 更新Top-K值: ${topkVal}`);
        }

        // 更新按钮状态
        updateTopkButtons();
    }
    // 添加input事件监听器，实现拖动时实时更新热力图(使用防抖优化)
    topkSlider.addEventListener('input', function () {
        updateTopkDisplayInternal();

        // 防抖: 延迟100ms更新热力图和统计信息
        clearTimeout(heatmapUpdateTimer);
        heatmapUpdateTimer = setTimeout(() => {
            if (filteredMatrix || globalUIState.dataSource.currentMatrix) {
                updateHeatmap();

                // 统计信息也使用防抖,在热力图更新后执行
                clearTimeout(statsUpdateTimer);
                statsUpdateTimer = setTimeout(() => {
                    if (globalUIState.dataSource.subtractIndex !== null) {
                        showDifferenceStatistics();
                    } else {
                        showStatistics();
                    }
                }, 50);
            }
        }, DEBOUNCE_DELAY);
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
        btn.addEventListener('click', function () {
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

/**
 * 初始化显示字段选择器
 */
function initDisplayFieldSelectors() {
    const xDisplayFieldEl = document.getElementById('xDisplayField');
    const yDisplayFieldEl = document.getElementById('yDisplayField');

    if (!xDisplayFieldEl || !yDisplayFieldEl) {
        console.log('[显示字段选择器] 元素不存在，跳过初始化');
        return;
    }

    // X轴显示字段选择器
    xDisplayFieldEl.addEventListener('change', function () {
        const xField = this.value;
        console.log(`[显示字段] X轴字段切换为: ${xField}`);

        // 更新全局状态
        globalUIState.displayFields.xField = xField;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.displayFields.xField = xField;
            console.log(`[独占模式] 保存X轴显示字段: ${xField}`);
        }

        // 重新生成X轴标签并更新热力图
        if (globalUIState.dataSource.currentXData.length > 0) {
            globalUIState.dataSource.currentXLabels = generateUniqueLabels(
                globalUIState.dataSource.currentXData,
                xField
            );
            updateHeatmapFromGlobalState(false);
        }
    });

    // Y轴显示字段选择器
    yDisplayFieldEl.addEventListener('change', function () {
        const yField = this.value;
        console.log(`[显示字段] Y轴字段切换为: ${yField}`);

        // 更新全局状态
        globalUIState.displayFields.yField = yField;

        // *** 只在独占模式下才保存到图表配置 ***
        if (globalUIState.exclusiveMode.active && globalUIState.exclusiveMode.editingIndex !== null) {
            const config = allSimilarityResults[globalUIState.exclusiveMode.editingIndex].visualConfig;
            config.displayFields.yField = yField;
            console.log(`[独占模式] 保存Y轴显示字段: ${yField}`);
        }

        // 重新生成Y轴标签并更新热力图
        if (globalUIState.dataSource.currentYData.length > 0) {
            globalUIState.dataSource.currentYLabels = generateUniqueLabels(
                globalUIState.dataSource.currentYData,
                yField
            );
            updateHeatmapFromGlobalState(false);
        }
    });
}

// =====
// 导出功能
// =====


// 导出JSON功能
async function exportToJSON() {
    if (!allSimilarityResults || allSimilarityResults.length === 0) {
        showError('没有可导出的数据，请先计算相似度');
        return;
    }

    try {
        // 显示导出状态
        const exportBtn = document.getElementById('exportJsonBtn');
        exportBtn.textContent = '导出中...';
        exportBtn.disabled = true;

        // 获取选中要导出的图表索引
        const exportSelect = document.getElementById('exportMatrixSelect');
        const selectedIndex = parseInt(exportSelect.value);

        if (isNaN(selectedIndex) || selectedIndex < 0 || selectedIndex >= allSimilarityResults.length) {
            showError('请选择要导出的图表');
            return;
        }

        // 获取选中的图表数据
        const selectedResult = allSimilarityResults[selectedIndex];

        // 构建导出数据结构
        // 注意: 这里导出的是 visualConfig 中的原始数值配置，而不是布尔矩阵
        const exportData = {
            version: "1.0",
            timestamp: new Date().toISOString(),
            exportedMatrix: {
                index: selectedIndex,
                xCollection: selectedResult.xCollection,
                yCollection: selectedResult.yCollection,
                matrix: selectedResult.matrix,  // 原始相似度矩阵
                xData: selectedResult.xData,
                yData: selectedResult.yData,
                xAvailableFields: selectedResult.xAvailableFields,
                yAvailableFields: selectedResult.yAvailableFields
                // stats字段已移除,统计信息现在实时计算
            },
            // 导出该图的可视化配置（包含原始的数值配置）
            visualConfig: {
                displayFields: {
                    xField: selectedResult.visualConfig.displayFields.xField,
                    yField: selectedResult.visualConfig.displayFields.yField
                },
                // 导出原始阈值配置（数值形式）
                similarityRange: {
                    min: selectedResult.visualConfig.similarityRange.min,
                    max: selectedResult.visualConfig.similarityRange.max
                },
                // 导出原始 Top-K 配置（数值形式）
                filters: {
                    topK: {
                        value: selectedResult.visualConfig.filters.topK.value,
                        axis: selectedResult.visualConfig.filters.topK.axis
                    }
                },
                // 导出排序配置
                sorting: {
                    order: selectedResult.visualConfig.sorting.order
                }
                // 注意: 不导出 cachedMasks (布尔矩阵缓存)，因为它们是计算中间结果
            }
        };

        // 转换为JSON字符串
        const jsonString = JSON.stringify(exportData, null, 2);

        // 生成文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `相似度分析_图${selectedIndex + 1}_${selectedResult.xCollection}_vs_${selectedResult.yCollection}_${timestamp}.json`;

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

// =====
// 导入功能
// =====

/**
 * 触发文件选择器
 */
function triggerImportJSON() {
    const fileInput = document.getElementById('importFileInput');
    if (fileInput) {
        // 重置文件输入,允许重复导入同一个文件
        fileInput.value = '';
        fileInput.click();
    }
}

/**
 * 处理导入的文件(支持多文件)
 */
async function handleImportFile(event) {
    const files = Array.from(event.target.files);
    if (!files || files.length === 0) {
        return;
    }

    // 检查所有文件类型
    const invalidFiles = files.filter(file => !file.name.endsWith('.json'));
    if (invalidFiles.length > 0) {
        showError(`以下文件不是JSON格式: ${invalidFiles.map(f => f.name).join(', ')}`);
        return;
    }

    try {
        // *** 记录导入前的图表数量，用于判断是否首次导入 ***
        const isFirstImport = allSimilarityResults.length === 0;

        // 用于收集导入结果
        const importResults = {
            success: [],
            failed: []
        };

        // 逐个处理文件
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const fileIndex = i + 1;
            const totalFiles = files.length;

            try {
                showLoading(true, `正在导入第 ${fileIndex}/${totalFiles} 个文件: ${file.name}...`);

                // 读取文件内容
                const fileContent = await readFileAsText(file);

                // 解析JSON
                let importedData;
                try {
                    importedData = JSON.parse(fileContent);
                } catch (e) {
                    throw new Error(`JSON格式错误: ${e.message}`);
                }

                // 验证数据格式
                const validationError = validateImportedData(importedData);
                if (validationError) {
                    throw new Error(`数据验证失败: ${validationError}`);
                }

                // 导入数据到allSimilarityResults
                importDataToResults(importedData);

                // 初始化新图表的按钮状态
                const newIndex = allSimilarityResults.length - 1;
                if (!matrixButtonStates[newIndex]) {
                    matrixButtonStates[newIndex] = {
                        index: newIndex,
                        applyData: false,
                        applyFilter: false,
                        exclusive: false
                    };
                }

                // 记录成功
                importResults.success.push({
                    fileName: file.name,
                    xCollection: importedData.exportedMatrix.xCollection,
                    yCollection: importedData.exportedMatrix.yCollection
                });

            } catch (error) {
                console.error(`导入文件 ${file.name} 时出错:`, error);
                importResults.failed.push({
                    fileName: file.name,
                    error: error.message
                });
            }
        }

        // 更新UI (使用新架构的函数)
        updateMatrixListUI();
        updateExportMatrixSelector();

        // 显示图表选择控制区域 (添加空值检查)
        const matrixSelectorControl = document.getElementById('matrixSelectorControl');
        const displayFieldControls = document.getElementById('displayFieldControls');
        const yDisplayFieldControls = document.getElementById('yDisplayFieldControls');
        const statsSection = document.getElementById('statsSection');
        const operationsSection = document.getElementById('operationsSection');

        if (matrixSelectorControl) matrixSelectorControl.style.display = 'block';
        if (displayFieldControls) displayFieldControls.style.display = 'block';
        if (yDisplayFieldControls) yDisplayFieldControls.style.display = 'block';
        if (statsSection) statsSection.style.display = 'block';
        if (operationsSection) operationsSection.style.display = 'block';

        // *** 首次导入数据时自动启用第一张图的独占模式 ***
        if (isFirstImport && allSimilarityResults.length > 0) {
            console.log('[自动独占] 首次导入数据，自动启用图0的独占模式');
            await enterExclusiveMode(0);
            updateMatrixListUI(); // 再次更新UI以反映独占模式状态
        }

        // 显示导入结果
        showLoading(false);

        let resultMessage = '';
        if (importResults.success.length > 0) {
            resultMessage += `成功导入 ${importResults.success.length} 个文件`;
            if (importResults.success.length === 1) {
                const item = importResults.success[0];
                resultMessage = `成功导入数据: ${item.xCollection} vs ${item.yCollection}`;
            }
            if (isFirstImport) {
                resultMessage += '，已自动启用编辑模式';
            }
        }

        if (importResults.failed.length > 0) {
            const failedList = importResults.failed.map(item =>
                `${item.fileName}: ${item.error}`
            ).join('\n');

            if (importResults.success.length > 0) {
                showError(`部分文件导入失败:\n${failedList}\n\n${resultMessage}`);
            } else {
                showError(`所有文件导入失败:\n${failedList}`);
            }
        } else {
            showSuccess(resultMessage);
        }

    } catch (error) {
        console.error('导入数据时出错:', error);
        showError('导入出错: ' + error.message);
        showLoading(false);
    }
}

/**
 * 读取文件为文本
 */
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = () => reject(new Error('文件读取失败'));
        reader.readAsText(file);
    });
}

/**
 * 验证导入的数据格式
 */
function validateImportedData(data) {
    // 检查版本
    if (!data.version) {
        return '缺少版本信息';
    }

    // 检查exportedMatrix
    if (!data.exportedMatrix) {
        return '缺少exportedMatrix字段';
    }

    const matrix = data.exportedMatrix;

    // 检查必要字段
    const requiredFields = [
        'xCollection', 'yCollection', 'matrix',
        'xData', 'yData',
        'xAvailableFields', 'yAvailableFields'
        // 注意: stats字段已改为实时计算,不再作为必需字段
    ];

    for (const field of requiredFields) {
        if (!matrix[field]) {
            return `缺少必要字段: ${field}`;
        }
    }

    // 检查数组类型
    if (!Array.isArray(matrix.matrix)) {
        return 'matrix必须是数组';
    }
    if (!Array.isArray(matrix.xData)) {
        return 'xData必须是数组';
    }
    if (!Array.isArray(matrix.yData)) {
        return 'yData必须是数组';
    }
    if (!Array.isArray(matrix.xAvailableFields)) {
        return 'xAvailableFields必须是数组';
    }
    if (!Array.isArray(matrix.yAvailableFields)) {
        return 'yAvailableFields必须是数组';
    }

    // 检查矩阵维度
    if (matrix.matrix.length !== matrix.yData.length) {
        return `矩阵行数(${matrix.matrix.length})与yData长度(${matrix.yData.length})不匹配`;
    }

    if (matrix.matrix.length > 0 && matrix.matrix[0].length !== matrix.xData.length) {
        return `矩阵列数(${matrix.matrix[0].length})与xData长度(${matrix.xData.length})不匹配`;
    }

    // 检查visualConfig (可选,如果不存在则使用默认值)
    if (data.visualConfig) {
        if (!data.visualConfig.displayFields) {
            return 'visualConfig缺少displayFields';
        }
    }

    return null; // 验证通过
}

/**
 * 将导入的数据添加到allSimilarityResults
 */
function importDataToResults(importedData) {
    const matrix = importedData.exportedMatrix;

    // 如果有visualConfig就使用,否则创建默认配置
    let visualConfig;
    if (importedData.visualConfig) {
        // 使用导入的配置,但需要补充cachedMasks
        visualConfig = {
            displayFields: {
                xField: importedData.visualConfig.displayFields.xField,
                yField: importedData.visualConfig.displayFields.yField
            },
            similarityRange: {
                min: importedData.visualConfig.similarityRange?.min ?? 0,
                max: importedData.visualConfig.similarityRange?.max ?? 1
            },
            filters: {
                topK: {
                    value: importedData.visualConfig.filters?.topK?.value ?? 0,
                    axis: importedData.visualConfig.filters?.topK?.axis ?? 'x'
                }
            },
            sorting: {
                order: importedData.visualConfig.sorting?.order ?? 'none'
            },
            cachedMasks: {
                thresholdMask: null,
                topKMask: null,
                finalMask: null
            }
        };
    } else {
        // 创建默认配置
        visualConfig = createDefaultVisualConfig(
            matrix.xAvailableFields,
            matrix.yAvailableFields
        );
    }

    // 构建结果对象并添加到数组
    const resultObject = {
        xCollection: matrix.xCollection,
        yCollection: matrix.yCollection,
        matrix: matrix.matrix,
        xData: matrix.xData,
        yData: matrix.yData,
        xAvailableFields: matrix.xAvailableFields,
        yAvailableFields: matrix.yAvailableFields,
        // stats字段已移除,统计信息现在实时计算
        visualConfig: visualConfig
    };

    allSimilarityResults.push(resultObject);
}

// =====
// 事件监听器
// =====

// 监听窗口大小变化，重新调整图表大小
window.addEventListener('resize', function () {
    if (document.getElementById('heatmap').innerHTML && filteredMatrix) {
        // 延迟执行以确保容器尺寸已更新
        setTimeout(() => {
            resizeHeatmap();
        }, 150);
    }
});

// =====
// 新架构：核心状态管理函数
// =====

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

    if (currentState) {
        // 当前已启用，要关闭
        // 如果当前是独占模式，先退出
        if (globalUIState.exclusiveMode.active) {
            exitExclusiveMode();
        }

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
        // 当前未启用，要开启 - 先检查矩阵大小一致性
        const sizeCheck = checkMatrixSizeConsistency(index, 'applyData');
        if (!sizeCheck.isValid) {
            console.warn(`[应用数据] 矩阵大小检查失败: ${sizeCheck.message}`);
            showWarning(sizeCheck.message, 4000);
            return; // 阻止按钮切换，不退出独占模式
        }

        // 矩阵大小检查通过后，才退出独占模式
        if (globalUIState.exclusiveMode.active) {
            exitExclusiveMode();
        }

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

        differenceMatrices[cacheKey] = matrix1.map((row, i) =>
            row.map((val, j) => val - matrix2[i][j])
        );
    }

    // 使用主数据源的元数据，但矩阵是差值
    const primaryData = allSimilarityResults[primaryIndex];

    globalUIState.dataSource.currentMatrix = differenceMatrices[cacheKey];
    globalUIState.dataSource.currentXData = primaryData.xData;
    globalUIState.dataSource.currentYData = primaryData.yData;
    globalUIState.dataSource.xAvailableFields = primaryData.xAvailableFields;
    globalUIState.dataSource.yAvailableFields = primaryData.yAvailableFields;

    // 更新显示字段（使用主图的配置）
    const primaryXField = primaryData.visualConfig.displayFields.xField;
    const primaryYField = primaryData.visualConfig.displayFields.yField;

    globalUIState.displayFields.xField = primaryXField;
    globalUIState.displayFields.yField = primaryYField;

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

    // *** 重置临时筛选器（差值模式）***
    resetTemporaryFilter(true);
    applyTemporaryFilterToUI();

    // *** 差值模式：将阈值重置为 -1 到 1 ***
    globalUIState.filters.uiState.similarityRange.min = -1;
    globalUIState.filters.uiState.similarityRange.max = 1;

    // 获取UI控件
    const minSlider = document.getElementById('minSimilaritySlider');
    const maxSlider = document.getElementById('maxSimilaritySlider');
    const minInput = document.getElementById('minSimilarityInput');
    const maxInput = document.getElementById('maxSimilarityInput');

    // *** 重要：先更新滑块的min/max范围，再设置value ***
    minSlider.min = -1;
    minSlider.max = 1;
    maxSlider.min = -1;
    maxSlider.max = 1;
    minInput.min = -1;
    minInput.max = 1;
    maxInput.min = -1;
    maxInput.max = 1;

    // 然后再设置滑块的值
    minSlider.value = -1;
    maxSlider.value = 1;
    minInput.value = -1;
    maxInput.value = 1;

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    const pos = calculateTrackPosition(-1, 1, minSlider);
    track.style.left = pos.left + 'px';
    track.style.width = pos.width + 'px';

    console.log('[差值模式] 阈值已重置为 -1.00 到 1.00');
}

/**
 * 切换"应用筛选器"按钮
 * @param {number} index - 图表索引
 */
function toggleApplyFilterButton(index) {
    console.log(`[应用筛选器] 切换按钮 ${index}`);

    const currentState = matrixButtonStates[index].applyFilter;

    if (currentState) {
        // 当前已启用，要关闭
        // 如果当前是独占模式，先退出
        if (globalUIState.exclusiveMode.active) {
            exitExclusiveMode();
        }

        matrixButtonStates[index].applyFilter = false;
        globalUIState.filters.activeFilterIndices = globalUIState.filters.activeFilterIndices.filter(i => i !== index);
    } else {
        // 当前未启用，要开启 - 先检查矩阵大小一致性
        const sizeCheck = checkMatrixSizeConsistency(index, 'applyFilter');
        if (!sizeCheck.isValid) {
            console.warn(`[应用筛选器] 矩阵大小检查失败: ${sizeCheck.message}`);
            showWarning(sizeCheck.message, 4000);
            return; // 阻止按钮切换，不退出独占模式
        }

        // 矩阵大小检查通过后，才退出独占模式
        if (globalUIState.exclusiveMode.active) {
            exitExclusiveMode();
        }

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
async function toggleExclusiveModeButton(index) {
    console.log(`[独占模式] 切换按钮 ${index}`);

    const currentState = matrixButtonStates[index].exclusive;

    if (currentState) {
        // 当前已是独占模式，退出
        exitExclusiveMode();
    } else {
        // 进入独占模式
        await enterExclusiveMode(index);
    }

    updateMatrixListUI();
}

/**
 * 进入独占模式
 * @param {number} index - 图表索引
 */
async function enterExclusiveMode(index) {
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

    // *** 重置临时筛选器（单图模式）***
    resetTemporaryFilter(false);
    applyTemporaryFilterToUI();

    // 应用该图的筛选器配置
    const config = allSimilarityResults[index].visualConfig;
    globalUIState.filters.uiState.similarityRange = { ...config.similarityRange };
    globalUIState.filters.uiState.topK = { ...config.filters.topK };
    globalUIState.sorting.order = config.sorting.order;

    // 更新UI
    applyFilterStateToUI();

    // 自动选中导出下拉框中的对应图表
    const exportSelect = document.getElementById('exportMatrixSelect');
    if (exportSelect) {
        exportSelect.value = index.toString();
    }

    // 显示提示
    showInfo('已进入独占编辑模式，您的修改将保存到此图的配置中', 3000);

    // *** 修复Bug: 等待热力图更新完成 ***
    await updateHeatmapFromGlobalState();
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

    // 更新滑块轨道
    const track = document.getElementById('similarityTrack');
    const pos = calculateTrackPosition(range.min, range.max, minSlider);
    track.style.left = pos.left + 'px';
    track.style.width = pos.width + 'px';

    // 更新Top-K
    document.getElementById('topkSlider').value = topK.value;
    currentTopkAxis = topK.axis;
    updateTopkDisplay();

    // 更新轴选择按钮
    document.getElementById('xAxisBtn').classList.toggle('active', topK.axis === 'x');
    document.getElementById('yAxisBtn').classList.toggle('active', topK.axis === 'y');
}


/**
 * 更新显示字段选择器
 */
function updateDisplayFieldSelectorsFromGlobalState() {
    const xFields = globalUIState.dataSource.xAvailableFields;
    const yFields = globalUIState.dataSource.yAvailableFields;

    // 显示控件（添加空值检查）
    const displayFieldControls = document.getElementById('displayFieldControls');
    const yDisplayFieldControls = document.getElementById('yDisplayFieldControls');

    if (displayFieldControls) displayFieldControls.style.display = 'block';
    if (yDisplayFieldControls) yDisplayFieldControls.style.display = 'block';

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

// 新增：显示差值统计信息
function showDifferenceStatistics(activatePanel = true) {
    const statsGrid = document.getElementById('statsGrid');

    if (!statsGrid) {
        console.warn('statsGrid元素不存在，统计信息面板可能未加载');
        return;
    }

    // 获取被减数(ground truth)和减数的布尔矩阵
    const primaryIndex = globalUIState.dataSource.primaryIndex;
    const subtractIndex = globalUIState.dataSource.subtractIndex;

    if (primaryIndex === null || subtractIndex === null) {
        statsGrid.innerHTML = '<div style="padding: 10px; text-align: center; color: #999;">差值模式数据不完整</div>';
        return;
    }

    // 获取被减数的最终遮罩(ground truth)
    const primaryConfig = allSimilarityResults[primaryIndex].visualConfig;
    const groundTruthMask = primaryConfig.cachedMasks?.finalMask || computeFinalMaskForMatrix(primaryIndex);

    // 获取当前应用的最终布尔矩阵
    let currentMask;

    // 如果启用了临时筛选器,应用临时筛选器
    const tempFilterMask = computeTemporaryFilterMask();
    if (tempFilterMask) {
        currentMask = tempFilterMask;
    } else {
        // 否则使用减数的最终遮罩
        const subtractConfig = allSimilarityResults[subtractIndex].visualConfig;
        currentMask = subtractConfig.cachedMasks?.finalMask || computeFinalMaskForMatrix(subtractIndex);
    }

    if (!groundTruthMask || !currentMask) {
        statsGrid.innerHTML = '<div style="padding: 10px; text-align: center; color: #999;">无法计算统计数据</div>';
        return;
    }

    // 计算差值模式统计信息
    const stats = calculateDifferenceModeStatistics(groundTruthMask, currentMask);

    // 构建HTML
    let statsHTML = '';

    // 1. True Positive
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-check-circle-fill text-success"></i></div>
            <div class="label">True Positive</div>
            <div class="value">${stats.truePositive}</div>
        </div>
    `;

    // 2. True Negative
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-circle text-secondary"></i></div>
            <div class="label">True Negative</div>
            <div class="value">${stats.trueNegative}</div>
        </div>
    `;

    // 3. False Positive
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-exclamation-circle-fill text-warning"></i></div>
            <div class="label">False Positive</div>
            <div class="value">${stats.falsePositive}</div>
        </div>
    `;

    // 4. False Negative
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-x-circle-fill text-danger"></i></div>
            <div class="label">False Negative</div>
            <div class="value">${stats.falseNegative}</div>
        </div>
    `;

    // 5. 上下文召回率
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-arrow-left-circle-fill"></i></div>
            <div class="label">上下文召回率</div>
            <div class="value">${(stats.contextRecall * 100).toFixed(2)}%</div>
        </div>
    `;

    // 6. 上下文精度
    statsHTML += `
        <div class="stat-item">
            <div class="icon"><i class="bi bi-crosshair2"></i></div>
            <div class="label">上下文精度</div>
            <div class="value">${(stats.contextPrecision * 100).toFixed(2)}%</div>
        </div>
    `;

    statsGrid.innerHTML = statsHTML;

    if (activatePanel) {
        // 激活统计信息按钮（使用新架构）
        const statisticsBtn = document.querySelector('[data-side="right"][data-position="top"][data-panel="statistics"]');
        if (statisticsBtn && !statisticsBtn.classList.contains('active')) {
            // 取消同位置其他按钮的激活状态
            if (activeButtons.right.top && activeButtons.right.top !== statisticsBtn) {
                activeButtons.right.top.classList.remove('active');
            }
            statisticsBtn.classList.add('active');
            activeButtons.right.top = statisticsBtn;

            // 关键：调用 updateSidebarContent 来移动面板
            updateSidebarContent('right', 'top', 'statistics');
        }

        // 激活筛选器控制按钮（使用新架构）
        const filtersBtn = document.querySelector('[data-side="right"][data-position="bottom"][data-panel="filters"]');
        if (filtersBtn && !filtersBtn.classList.contains('active')) {
            // 取消同位置其他按钮的激活状态
            if (activeButtons.right.bottom && activeButtons.right.bottom !== filtersBtn) {
                activeButtons.right.bottom.classList.remove('active');
            }
            filtersBtn.classList.add('active');
            activeButtons.right.bottom = filtersBtn;

            // 关键：调用 updateSidebarContent 来移动面板
            updateSidebarContent('right', 'bottom', 'filters');
        }

        // 更新右侧边栏状态（打开侧边栏）
        updateSidebarState('right');
    }
}



/**
 * 从全局状态更新热力图
 */
async function updateHeatmapFromGlobalState(activatePanel = true) {
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

    // *** 修复Bug1: 先显示统计信息和激活右侧栏，再绘制热力图 ***
    if (globalUIState.dataSource.primaryIndex !== null) {
        if (globalUIState.dataSource.subtractIndex !== null) {
            showDifferenceStatistics(activatePanel);
        } else {
            showStatistics(activatePanel);
        }
    }

    // 等待DOM更新完成（右侧栏动画完成）
    await new Promise(resolve => setTimeout(resolve, 50));

    // 创建或更新热力图（此时容器大小已经是正确的）
    const isFirstTimeCreating = document.getElementById('heatmap').innerHTML === '';
    if (isFirstTimeCreating) {
        createHeatmap();
        // *** 修复Bug: 首次创建热力图后，立即调用updateHeatmap以应用筛选器 ***
        // createHeatmap()只是绘制完整图形，筛选器需要通过updateHeatmap()应用
        console.log('[热力图] 首次创建完成，立即应用筛选器');
        updateHeatmap();
    } else {
        updateHeatmap();
    }

    // 显示操作区域 - 面板会在showStatistics中自动打开
    // 不需要显式控制section的display，因为它们在面板内部
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

        // 确定边框颜色和状态图标
        let borderClass = '';
        let badgeIcon;
        let badgeClass;

        if (state.exclusive) {
            // 独占模式 - 紫色边框 + 准星角标
            borderClass = 'border-exclusive';
            badgeIcon = 'bi-crosshair';
            badgeClass = 'badge-crosshair';
        } else if (globalUIState.dataSource.primaryIndex === index && globalUIState.dataSource.subtractIndex === null) {
            // 主数据源 - 绿色边框 + 实心圆圈角标
            borderClass = 'border-primary2';
            badgeIcon = 'bi-circle-fill';
            badgeClass = 'badge-circle';
        } else if (globalUIState.dataSource.primaryIndex === index && globalUIState.dataSource.subtractIndex !== null) {
            // 被减数 - 绿色边框 + 实心加号角标
            borderClass = 'border-primary2';
            badgeIcon = 'bi-plus-circle-fill';
            badgeClass = 'badge-plus';
        } else if (globalUIState.dataSource.subtractIndex === index) {
            // 减数 - 红色边框 + 实心减号角标
            borderClass = 'border-subtract';
            badgeIcon = 'bi-dash-circle-fill';
            badgeClass = 'badge-minus';
        } else {
            // 无状态 - 默认边框 + 灰色方形角标
            badgeIcon = 'bi-bookmark';
            badgeClass = 'badge-none';
        }

        if (borderClass) row.classList.add(borderClass);

        // 创建header容器(包含状态图标、标题和按钮)
        const header = document.createElement('div');
        header.className = 'matrix-item-header';

        // 左侧容器(状态图标 + 标题)
        const leftContainer = document.createElement('div');
        leftContainer.className = 'matrix-item-left';

        // 状态图标
        const badge = document.createElement('i');
        badge.className = `matrix-item-badge ${badgeClass} ${badgeIcon}`;
        leftContainer.appendChild(badge);

        // 图表标题 - 三行结构
        const title = document.createElement('div');
        title.className = 'matrix-item-title';

        const xName = document.createElement('div');
        xName.className = 'collection-name';
        xName.textContent = truncateCollectionName(result.xCollection, 20);
        xName.title = result.xCollection; // 完整名称作为tooltip

        const vsText = document.createElement('div');
        vsText.className = 'vs-text';
        vsText.textContent = 'vs';

        const yName = document.createElement('div');
        yName.className = 'collection-name';
        yName.textContent = truncateCollectionName(result.yCollection, 20);
        yName.title = result.yCollection; // 完整名称作为tooltip

        title.appendChild(xName);
        title.appendChild(vsText);
        title.appendChild(yName);

        leftContainer.appendChild(title);

        // 按钮容器
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'matrix-item-buttons';

        // 按钮1：应用数据 - 实心眼睛图标
        const dataBtn = document.createElement('button');
        dataBtn.className = 'matrix-control-btn btn-apply-data';
        dataBtn.innerHTML = '<i class="bi-eye-fill"></i>';
        dataBtn.title = '应用数据';
        if (state.applyData) dataBtn.classList.add('active');
        dataBtn.onclick = () => toggleApplyDataButton(index);

        // 按钮2：应用筛选器 - 筛选图标
        const filterBtn = document.createElement('button');
        filterBtn.className = 'matrix-control-btn btn-apply-filter';
        filterBtn.innerHTML = '<i class="bi-funnel-fill"></i>';
        filterBtn.title = '应用筛选器';
        if (state.applyFilter) filterBtn.classList.add('active');
        filterBtn.onclick = () => toggleApplyFilterButton(index);

        // 按钮3：独占模式 - 准星图标
        const exclusiveBtn = document.createElement('button');
        exclusiveBtn.className = 'matrix-control-btn btn-exclusive';
        exclusiveBtn.innerHTML = '<i class="bi-crosshair"></i>';
        exclusiveBtn.title = '独占模式';
        if (state.exclusive) exclusiveBtn.classList.add('active');
        exclusiveBtn.onclick = () => toggleExclusiveModeButton(index);

        // 组装
        buttonsContainer.appendChild(dataBtn);
        buttonsContainer.appendChild(filterBtn);
        buttonsContainer.appendChild(exclusiveBtn);

        header.appendChild(leftContainer);
        header.appendChild(buttonsContainer);

        row.appendChild(header);
        container.appendChild(row);
    });
}

/**
 * 更新导出图表选择器
 */
function updateExportMatrixSelector() {
    const select = document.getElementById('exportMatrixSelect');
    if (!select) return;

    // 清空现有选项
    select.innerHTML = '';

    if (!allSimilarityResults || allSimilarityResults.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '请先计算相似度';
        select.appendChild(option);
        return;
    }

    // 为每个图表添加选项
    allSimilarityResults.forEach((result, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `图 ${index + 1}: ${result.xCollection} vs ${result.yCollection}`;
        select.appendChild(option);
    });

    // 默认选中第一个
    if (allSimilarityResults.length > 0) {
        select.value = '0';
    }
}

// 页面加载时初始化
// 注意: 大部分控件初始化已经移到rebindEventHandlers中,在侧边栏内容加载时才执行
document.addEventListener('DOMContentLoaded', function () {
    // 仅初始化全局控件(不在侧边栏中的)
    initColorSchemeSelector();

    // 注意: 以下初始化已移到rebindEventHandlers中:
    // - initRangeSlider() -> 在'filters'面板加载时
    // - initTopkSlider() -> 在'filters'面板加载时
    // - initSortSelector() -> 在'chartControl'面板加载时
    // - initDisplayFieldSelectors() -> 在'chartControl'面板加载时
    // - loadCollections() -> 在'dataSource'面板加载时
});
