# HPO-3D-Match: 基于数据驱动的点云位姿估计算法超参数优化框架

面向工业 3D 视觉与机器人抓取的表面匹配（Surface Matching）黑箱超参数优化框架，配套博士学位论文第三章《数据驱动的点云位姿估计算法参数离线优化方法》。

---

## 1. 核心特性

- **三大典型几何工件全覆盖**：
  - `bracket_planar`：完全非对称、2mm 超薄钣金工件；
  - `screw_black`：连续轴对称、微小尺度金属工件；
  - `star`：12 重离散高阶旋转对称、每 30° 物理等价手柄工件。
- **高保真点云预处理加速**：
  - 2D 各向同性 Grid $3\times 3$ 空间降采样与相机内参同步缩放；
  - 托盘物理工作区深度滤波（$Z \in [0.20\,\text{m}, 0.95\,\text{m}]$），完全无损保留薄片工件几何同时滤除 100% 远景地面噪点；
  - 内存单次常驻初始化，单帧匹配加速 10.1 倍（365ms $\to$ 36.2ms）。
- **双目标函数体系 (全英文自解释命名)**：
  - **分层字典序召回优先目标函数** (`lexicographical-recall-first`, 简称 `lexrecall`)：论文提出方法，基于二分图严格一对一最大匹配与动态大权重，彻底杜绝漏检逃逸；
  - **固定惩罚线性加权目标函数** (`fixed-penalty-baseline`, 简称 `fixedpen`)：经典常数标量化惩罚基准，用于消融对照。
- **三大流派核心采样算法**：支持 `TPE` (贝叶斯), `CMA-ES` (进化策略), `Random` (统计基准)；
- **剪枝策略算力加速消融**：支持 `NopPruner` (全量), `MedianPruner` (中位数早停)；
- **全参数显式自解释目录结构**：`results/{model}_{sampler}_{pruner}_{objective}_b{budget}_s{seed}/`。

---

## 2. 环境配置

```powershell
uv sync --locked
```
- Python 运行环境与所有依赖库通过 `pyproject.toml` 和 `uv.lock` 严格版本锁定；
- 底层依赖 HALCON 运行环境与 Python 绑定。

---

## 3. 标准数据清单与可视化

```powershell
# 1. 交互式点云滤波与真值可见性查看器
uv run --locked python visualize_pointcloud.py --scene-id 0 --image-id 62
start visualizations/pointcloud_interactive_viewer.html
```

---

## 4. 实验流程执行指南 (Scene 0)

### 阶段一：三大典型工件 500 轮主实验 (多对称性几何泛化)
```powershell
# 1. 薄板工件 (bracket_planar)
uv run --locked python run_experiment.py `
  --model bracket_planar `
  --sampler TPE `
  --pruner Nop `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42

# 2. 黑色螺丝 (screw_black)
uv run --locked python run_experiment.py `
  --model screw_black `
  --sampler TPE `
  --pruner Nop `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42

# 3. 星形手柄 (star)
uv run --locked python run_experiment.py `
  --model star `
  --sampler TPE `
  --pruner Nop `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42
```

### 阶段二：三大流派采样算法横向对比 (验证贝叶斯优化优越性)
```powershell
# CMA-ES (进化策略)
uv run --locked python run_experiment.py `
  --model bracket_planar `
  --sampler CmaEs `
  --pruner Nop `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42

# Random Search (无信息随机基线)
uv run --locked python run_experiment.py `
  --model bracket_planar `
  --sampler Random `
  --pruner Nop `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42
```

### 阶段三：目标函数形态消融研究 (Objective Function Ablation)
```powershell
# 固定惩罚线性加权目标函数 (Fixed-Penalty Baseline)
uv run --locked python run_experiment.py `
  --model bracket_planar `
  --sampler TPE `
  --pruner Nop `
  --budget 500 `
  --objective-version fixed-penalty-baseline `
  --seed 42
```

### 阶段四：剪枝策略消融研究 (Pruning Ablation)
```powershell
# MedianPruner 中位数剪枝
uv run --locked python run_experiment.py `
  --model bracket_planar `
  --sampler TPE `
  --pruner Median `
  --budget 500 `
  --objective-version lexicographical-recall-first `
  --seed 42
```

---

## 5. 结果汇总与大表生成

```powershell
uv run --locked python summarize_results.py
```

---

## 6. 自动化测试套件

```powershell
uv run --locked python -m unittest discover -s tests -p "test_*.py"
```
