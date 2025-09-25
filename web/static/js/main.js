// 全局变量
let currentPage = 'dashboard';

// DOM 加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    setupEventListeners();

    // 如果是首页，默认显示dashboard内容
    if (window.location.pathname === '/web/dashboard') {
        loadDashboardContent();
    }
});

// 初始化页面
function initializePage() {
    // 设置当前页面状态
    updateActiveNavigation();
}

// 设置事件监听器
function setupEventListeners() {
    // 导航按钮点击事件
    const navButtons = document.querySelectorAll('.nav-button[data-page]');
    navButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const page = this.getAttribute('data-page');
            loadPage(page);
        });
    });

    // 窗口大小变化事件（保留用于可能的响应式优化）
    window.addEventListener('resize', handleWindowResize);
}

// 处理窗口大小变化（保留但简化）
function handleWindowResize() {
    // 可以在这里添加响应式相关的处理逻辑
}

// 加载页面内容
async function loadPage(page) {
    const pageContent = document.getElementById('pageContent');

    // 显示加载状态
    showLoadingState();

    try {
        // 发送AJAX请求获取页面内容
        const response = await fetch(`/web/${page}`, {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (response.ok) {
            const html = await response.text();

            // 解析HTML，提取内容部分
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const content = doc.querySelector('#pageContent') || doc.querySelector('.container-fluid');

            if (content) {
                // 添加页面切换动画
                pageContent.classList.add('page-enter');
                pageContent.innerHTML = content.innerHTML;

                setTimeout(() => {
                    pageContent.classList.remove('page-enter');
                    pageContent.classList.add('page-enter-active');
                }, 10);

                setTimeout(() => {
                    pageContent.classList.remove('page-enter-active');
                }, 310);

                // 更新当前页面状态
                currentPage = page;
                updateActiveNavigation();

                // 更新浏览器历史记录
                history.pushState({page: page}, '', `/web/${page}`);

            } else {
                throw new Error('无法解析页面内容');
            }
        } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
    } catch (error) {
        console.error('加载页面失败:', error);
        showErrorState(error.message);
    }
}

// 显示加载状态
function showLoadingState() {
    const pageContent = document.getElementById('pageContent');
    pageContent.innerHTML = `
        <div class="d-flex justify-content-center align-items-center" style="min-height: 300px;">
            <div class="text-center">
                <div class="loading mb-3"></div>
                <p class="text-muted">加载中...</p>
            </div>
        </div>
    `;
}

// 显示错误状态
function showErrorState(message) {
    const pageContent = document.getElementById('pageContent');
    pageContent.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <h4 class="alert-heading">加载失败</h4>
            <p>${message}</p>
            <hr>
            <p class="mb-0">请稍后重试或联系管理员。</p>
        </div>
    `;
}

// 更新当前激活的导航项
function updateActiveNavigation() {
    // 移除所有激活状态
    const navButtons = document.querySelectorAll('.nav-button');
    navButtons.forEach(button => {
        button.classList.remove('active');
    });

    // 设置当前页面为激活状态
    const activeButton = document.querySelector(`[data-page="${currentPage}"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

// 加载仪表板内容
function loadDashboardContent() {
    const pageContent = document.getElementById('pageContent');
    pageContent.innerHTML = `
        <style>
/* 卡片样式 */
.card {
    border: none;
    background-color: var(--white);
    box-shadow: var(--shadow);
    transition: all 0.2s ease;
    border-radius: 0.75rem;
}

.card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.card-header {
    background-color: var(--white);
    border-bottom: 1px solid var(--gray-200);
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem 0.75rem 0 0 !important;
}

.card-header.bg-primary-new {
    background: linear-gradient(135deg, var(--primary-gradient-start), var(--primary-gradient-end)) !important;
    border-bottom: none;
    color: var(--white);
}

.card-header.bg-success-new {
    background: linear-gradient(135deg, var(--primary-gradient-start), var(--primary-gradient-end)) !important;
    border-bottom: none;
    color: var(--white);
}

.card-body {
    padding: 1.5rem;
}

.card-title {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 0;
}

.card-header.bg-primary-new .card-title,
.card-header.bg-success-new .card-title {
    color: var(--white);
}

/* 主要功能卡片容器 - 设置最大宽度 */
.main-cards-container {
    max-width: 900px;
    margin: 0 auto;
}

/* 系统状态卡片 - 与主要功能卡片对齐 */
.status-card-container {
    max-width: 900px;
    margin: 0 auto;
}

/* 状态指示样式 */
.bg-light-new {
    background-color: var(--gray-100) !important;
    border-radius: 0.75rem;
    padding: 1.5rem !important;
    transition: all 0.2s ease;
}

.bg-light-new:hover {
    background: linear-gradient(135deg, var(--secondary-gradient-start), var(--secondary-gradient-end)) !important;
    color: var(--white);
}
</style>


        <div class="row">
            <div class="col-12">
                <h1 class="h2 mb-4 text-center">欢迎使用 Kumi</h1>
                <p class="lead text-muted mb-5 text-center">知识库与大模型评测平台</p>

                <div class="main-cards-container">
                    <div class="row justify-content-center">
                        <div class="col-lg-6 col-md-6 mb-4">
                            <div class="card h-100">
                                <div class="card-header bg-primary-new text-white">
                                    <h5 class="card-title mb-0 text-center">
                                        <i class="bi bi-book me-2"></i>知识库管理
                                    </h5>
                                </div>
                                <div class="card-body text-center">
                                    <p class="card-text">管理你的知识库，包括连接、设置嵌入模型、上传文件和测试功能。</p>
                                    <a href="#" class="btn btn-primary nav-button" data-page="knowledge">开始使用</a>
                                </div>
                            </div>
                        </div>
                        <div class="col-lg-6 col-md-6 mb-4">
                            <div class="card h-100">
                                <div class="card-header bg-primary-new text-white">
                                    <h5 class="card-title mb-0 text-center">
                                        <i class="bi bi-cpu me-2"></i>大模型评测
                                    </h5>
                                </div>
                                <div class="card-body text-center">
                                    <p class="card-text">配置和测试大语言模型，上传测评数据集，执行评测任务。</p>
                                    <a href="#" class="btn btn-primary nav-button" data-page="llm/config">开始使用</a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="status-card-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0 text-center">
                                    <i class="bi bi-graph-up me-2"></i>系统状态
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3 col-sm-6 text-center mb-3">
                                        <div class="bg-light-new rounded p-3">
                                            <i class="bi bi-database text-primary" style="font-size: 2rem;"></i>
                                            <h6 class="mt-2 mb-0">知识库</h6>
                                            <small class="text-muted">运行正常</small>
                                        </div>
                                    </div>
                                    <div class="col-md-3 col-sm-6 text-center mb-3">
                                        <div class="bg-light-new rounded p-3">
                                            <i class="bi bi-cpu text-success" style="font-size: 2rem;"></i>
                                            <h6 class="mt-2 mb-0">大模型</h6>
                                            <small class="text-muted">运行正常</small>
                                        </div>
                                    </div>
                                    <div class="col-md-3 col-sm-6 text-center mb-3">
                                        <div class="bg-light-new rounded p-3">
                                            <i class="bi bi-cloud-upload text-info" style="font-size: 2rem;"></i>
                                            <h6 class="mt-2 mb-0">文件服务</h6>
                                            <small class="text-muted">运行正常</small>
                                        </div>
                                    </div>
                                    <div class="col-md-3 col-sm-6 text-center mb-3">
                                        <div class="bg-light-new rounded p-3">
                                            <i class="bi bi-speedometer2 text-warning" style="font-size: 2rem;"></i>
                                            <h6 class="mt-2 mb-0">评测服务</h6>
                                            <small class="text-muted">运行正常</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 绑定仪表板中的链接事件
    const dashboardLinks = document.querySelectorAll('#pageContent [data-page]');
    dashboardLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const page = this.getAttribute('data-page');
            loadPage(page);
        });
    });
}

// 处理浏览器后退/前进按钮
window.addEventListener('popstate', function(event) {
    if (event.state && event.state.page) {
        loadPage(event.state.page);
    }
});

// 在 main.js 中添加或更新
document.addEventListener('DOMContentLoaded', function() {
    // 导航按钮点击处理
    document.querySelectorAll('.nav-button').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const page = this.getAttribute('data-page');
            if (page) {
                window.location.href = `/web/${page}`;
            }
        });
    });
});
