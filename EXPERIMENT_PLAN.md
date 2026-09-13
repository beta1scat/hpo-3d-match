# 3D 点云位姿估计超参数优化实验方案规范 (EXPERIMENT_PLAN.md)

**活动实验协议 ID**：`hpo-3d-match-scene0-matrix-v2`  
**标准配置文件**：`configs/hpo_protocol.json`  
**合成域训练清单**：`data/manifests/itodd_scene0_v1/bop_manifest.csv` (Scene 0 PBR, 8:2 Train/Dev)  
**真实域标定清单**：`data/manifests/itodd_manual_annotated/bop_manifest.csv` (最新 19 场景物理传感器标定真实数据集)  
**原生工业场景清单**：`data/manifests/itodd_external/itodd_external_manifest.csv` (data/3d_long_baseline 原生工业点云)  
**统一实验执行引擎**：`run_experiment.py` / `main.py`  
**离线点云确定性滤波工具**：`preprocess_filter_tabletop.py` (固化纯工件点云至 points_tabletop_filtered/*.ply)  
**结果规整与汇总工具**：`summarize_results.py`  
**论文标准 4K 可视化工具**：`visualize_results.py` (生成原生工业场景 2D/3D 渲染图件)  
**出版级定性对比图组装工具**：`plotting_scripts/plot_qualitative_comparison.py` (4行x5列 600 DPI 矢量/超清对比大图)  
**全章实验图件一键生成流水线**：`plotting_scripts/generate_all_figures.py` (图 3-3 至图 3-7 全量自动绘制)  
**对应论文章节**：博士学位论文第三章《数据驱动的点云位姿估计算法参数离线优化方法》

---

## 1. 实验总体设计原则与工件典型性论证

本实验方案针对工业机器人 3D 视觉无序抓取场景中传统经验参数调优的局限性，在 BOP ITODD 工业料框（Scene 0）复杂点云数据上，系统性验证基于贝叶斯优化的离线参数自动寻优方法。

### 1.1 三大典型工件的几何拓扑对称性全光谱覆盖

| 工件名称与模型 ID | 物理外形与几何尺寸 | 对称群数学分类 | 3D 点云位姿估计核心难点机理 | 博士论文代表性价值 |
| :--- | :--- | :--- | :--- | :--- |
| **`bracket_planar`**<br>(obj_id=5) | $118 \times 118 \times \mathbf{2\,\text{mm}}$<br>(超薄金属冲压件) | **完全非对称**<br>($C_1$ 纯刚体拓扑) | **严重的 3D 孔径效应（Aperture Problem）**：内部点法向量平行，2mm 薄边在空间降采样中极易被抹除，传统默认参数检出为 0。 | 代表工业界最棘手的**钣金件与极薄壁结构件**无序抓取。 |
| **`screw_black`**<br>(obj_id=24) | $\varnothing 30 \times 60\,\text{mm}$<br>(黑色六角紧固螺栓) | **连续轴对称**<br>($SO(2)$ 李群对称) | **$Z$ 轴方向旋转自由度退化**：几何上绕对称轴旋转不可观，必须在商空间 $SE(3) / SO(2)$ 下进行无偏度量。 | 代表装配产线中最普遍的**轴类、销类与螺纹紧固件**。 |
| **`star`**<br>(obj_id=25) | $\varnothing 48.5 \times 5.7\,\text{mm}$<br>(12 齿星形调节把手) | **高阶离散对称**<br>($C_{12}$ 旋转群，每 $30^\circ$) | **多局部极小值陷阱**：在 $SO(3)$ 流形内存在 12 个等价能量极小值，位姿度量需执行严格的群对称模运算。 | 代表**齿轮、花键、法兰盘与多齿旋转把手**。 |

### 1.2 数据集划分与隔离学术规范

1. **合成域寻优（Sim HPO）**：基于 `data/manifests/itodd_scene0_v1/bop_manifest.csv`，严格按 8:2 隔离为 `train`（寻优）与 `dev`（验证选模）。
2. **真实域理论上限寻优（Real-to-Real Oracle）**：完全独立于合成数据，直接在最新的 19 场景真实工业手动标定数据集 `data/manifests/itodd_manual_annotated/bop_manifest.csv` 上运行大规模 HPO 寻优，结合各工件物理标定阈值的自适应 RANSAC 桌面点云剔除算子，建立物理真实域参数优化的理论性能天花板（Real-to-Real）。原 `itoddmv_val` 实验废止。
3. **合成与真实分离评价范式**：放弃效果受限的跨域 Sim-to-Real 零样本迁移方案，采取合成域（Sim）与真实域（Real）各自独立寻优、独立评价的实验范式，分别系统论证算法在仿真域与物理真实域的寻优收敛性与工业适用性。
4. **原生场景定性验证（Native Validation）**：在无 GT 标注的原生 ITODD 工业点云（`data/3d_long_baseline`）上运行固定前向推理，进行全视角 4K 超高清定性可视化展示。

---

## 2. 9 维纯物理超参数搜索空间与固定常数定义

| 参数名称 | 参数类型 | 官方默认值 | 官方推荐区间 | 全景优化搜索空间 | 物理调控目标与机理 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`RelSamplingDistance`** | 连续浮点 | `0.05` | $[0.03, 0.10]$ | **`[0.010, 0.20]`** | 几何特征点云采样间距（薄板需极细下探） |
| **`KeyPointFraction`** | 连续浮点 | `0.20` | $[0.05, 0.30]$ | **`[0.010, 0.60]`** | 参与 PPF 特征点对配对的关键点密度比例 |
| **`min_score`** | 连续浮点 | `0.00` | $[0.00, 0.40]$ | **`[0.000, 0.70]`** | 归一化可见表面点覆盖率截断阈值 |
| **`max_overlap_dist_rel`**| 连续浮点 | `0.50` | $[0.10, 1.00]$ | **`[0.050, 1.00]`** | 非极大值抑制（NMS）空间重叠抑制半径 |
| **`pose_ref_num_steps`** | 离散整数 | `5` | $[1, 20]$ | **`[1, 50]`** | ICP 姿态精修迭代步数（直接控制精度与耗时） |
| **`pose_ref_sub_sampling`**| 离散整数 | `2` | $[1, 10]$ | **`[1, 25]`** | ICP 场景点云稀疏降采样比例（加速收敛） |
| **`pose_ref_dist_threshold_rel`**| 连续浮点| `0.10` | $[0.03, 0.20]$ | **`[0.005, 0.50]`** | ICP 点对关联的最大欧氏截断距离容差 |
| **`pose_ref_scoring_dist_rel`**| 连续浮点| `0.005`| $[0.0001, 0.20]$| **`[0.0001, 0.25]`**| 判定点落入模型表面的最终评分容差 |
| **`pose_ref_use_scene_normals`**| 类别布尔| `'false'`| `['true', 'false']`| **`['true', 'false']`**| ICP 精修是否引入法向量方向一致性约束 |

### 4 个底层固定常数与硬熔断机制：
- **`scene_normal_computation` = `'fast'`**：彻底消除 MLS 导致的 2 分钟异常预处理阻塞；
- **`dense_pose_refinement` = `'true'`**：确保 ICP 精修参数 100% 生效，杜绝平坦鞍区；
- **`sparse_pose_refinement` = `'true'`**：保持粗-精两级标准工业配准流水线；
- **`score_type` = `'model_point_fraction'`**：保持置信度与 `min_score` 物理语义严格对齐；
- **`timeout_sec` = `0.5`**：引入 $0.5\,\text{s}$ 算子级硬熔断机制，杜绝病态参数耗尽算力。

---

## 3. 实验目录命名规范

$$\mathbf{results/\{model\}\_\{sampler\}\_\{pruner\}\_\{objective\}\_b\{budget\}\_s\{seed\}/}$$

- **`model`**：`bracket_planar` / `screw_black` / `star`
- **`sampler`**：`tpe`（非参数贝叶斯）/ `cmaes`（协方差自适应进化）/ `random`（统计无信息基线）
- **`pruner`**：`nop`（无剪枝基准）/ `median`（中位数早停）
- **`objective`**：`lexrecall`（全局默认：字典序召回优先）/ `fixedpen`（固定惩罚消融基准）
- **`budget`**：统一设定为 `b500`（500 轮全量寻优，每 50 轮执行一次全量 Dev 评估）
- **`seed`**：统一设定为 `s42`

---

## 4. 博士论文第三章完整实验执行指令集 (Powershell)

### 阶段零：官方默认超参数定量基线评测 (Default Baseline Quantitative Evaluation)
在执行大规模 HPO 寻优前，独立评测未优化状态下的算法性能（用于支撑 Table 3.4 与 Table 3.7 的 Default 基线对比）：

```powershell
# 0. 真实域点云确定性去桌面离线预处理 (固化 19 个真实标定场景工件点云至 points_tabletop_filtered/*.ply，彻底杜绝动态 RANSAC 随机扰动)
uv run python preprocess_filter_tabletop.py

# 1. 合成域 Dev 集默认参数评测 (Scene 0 Dev，用于 Table 3.4 基线)
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model bracket_planar --split dev --results-root results/baseline_default_scene0/bracket_planar --run-id eval-default-dev
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model screw_black --split dev --results-root results/baseline_default_scene0/screw_black --run-id eval-default-dev
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model star --split dev --results-root results/baseline_default_scene0/star --run-id eval-default-dev

# 2. 真实域手动标注数据集默认参数评测 (启用 RANSAC 桌面点云剔除预处理，用于 Table 3.7 真实基线)
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model bracket_planar --split dev --filter-tabletop --results-root results/baseline_default_manual/bracket_planar --run-id eval-default-manual-ransac
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model screw_black --split dev --filter-tabletop --results-root results/baseline_default_manual/screw_black --run-id eval-default-manual-ransac
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model star --split dev --filter-tabletop --results-root results/baseline_default_manual/star --run-id eval-default-manual-ransac
```

---

### 阶段一：$3 \times 3$ 核心矩阵大比武（主实验，共 9 组）
```powershell
# 终端 1：薄板工件 bracket_planar (3 算法 x 500 轮)
uv run python run_experiment.py --model bracket_planar --matrix --budget 500 --pruner Nop --seed 42

# 终端 2：黑色螺丝 screw_black (3 算法 x 500 轮)
uv run python run_experiment.py --model screw_black --matrix --budget 500 --pruner Nop --seed 42

# 终端 3：星形手柄 star (3 算法 x 500 轮)
uv run python run_experiment.py --model star --matrix --budget 500 --pruner Nop --seed 42
```
> **论文对应成果**：
> - **Table 3.4**：《三大流派采样算法在三类典型对称性工件上的位姿估计性能全矩阵对比表》；
> - **Figure 3.4**：《不同对称性工件下 TPE / CMA-ES / Random 在 500 轮内的全局收敛速度与轨迹对比图》。

---

### 阶段二：目标函数构造形态消融研究（Objective Function Ablation）
```powershell
# 运行传统固定惩罚基准 (Fixed-Penalty Baseline，用于与阶段一的 LexRecall 对比)
uv run python run_experiment.py --model bracket_planar --sampler TPE --pruner Nop --budget 500 --objective-version fixed-penalty-baseline --seed 42
uv run python run_experiment.py --model screw_black --sampler TPE --pruner Nop --budget 500 --objective-version fixed-penalty-baseline --seed 42
uv run python run_experiment.py --model star --sampler TPE --pruner Nop --budget 500 --objective-version fixed-penalty-baseline --seed 42
```
> **论文对应成果**：**Table 3.5**：《目标函数构造形式对检出率抑制与位姿精度的消融对比表》。

---

### 阶段三：剪枝策略算力加速消融研究（Pruning Acceleration Ablation）
```powershell
# 运行中位数早停剪枝 (MedianPruner，用于与阶段一的无剪枝 Nop 对比)
uv run python run_experiment.py --model bracket_planar --sampler TPE --pruner Median --budget 500 --seed 42
uv run python run_experiment.py --model screw_black --sampler TPE --pruner Median --budget 500 --seed 42
uv run python run_experiment.py --model star --sampler TPE --pruner Median --budget 500 --seed 42
```
> **论文对应成果**：**Table 3.6**：《剪枝策略在搜索总耗时、剪枝触发率与最终位姿精度上的加速消融对比表》。

---

### 阶段四：真实域理论上限寻优（Real-to-Real Oracle 1000 轮极限寻优）

放弃效果受限的跨域 Sim-to-Real 迁移方案，直接在最新的 19 场景真实工业手动标定数据集 `data/manifests/itodd_manual_annotated/bop_manifest.csv` 上开展大规模 Real-to-Real HPO 极限寻优，建立真实工业场景下的理论物理性能天花板（原 `itoddmv_val` 实验全面作废）。

通过显式指定 `--results-root`，结果保存至独立的真实域目录，系统自动识别 `itodd_manual_annotated` 真实数据集并开启 **桌面点云 RANSAC 滤波预处理**（star: 1.0mm, bracket: 0.8mm, screw_black: 4.0mm，剔除料框桌面冗余点，大幅加速匹配收敛），自动均匀记录 10 个 Checkpoints：

```powershell
# 薄板 bracket_planar 真实域 1000 轮 Oracle 寻优 (19 场景，物理阈值 0.8mm)
uv run python run_experiment.py --model bracket_planar --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_bracket_planar_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42

# 黑色螺丝 screw_black 真实域 1000 轮 Oracle 寻优 (19 场景，物理阈值 4.0mm)
uv run python run_experiment.py --model screw_black --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_screw_black_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42

# 星形调节把手 star 真实域 1000 轮 Oracle 寻优 (19 场景，物理阈值 1.0mm)
uv run python run_experiment.py --model star --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_star_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42
```
> **论文对应成果**：
> - **Table 3.7**：《物理真实工业场景默认基线与 Real-to-Real 理论性能天花板对比表》；
> - **Figure 3.7**：《真实工业场景下 1000 轮贝叶斯参数优化演进轨迹与收敛曲线图》。

---

### 阶段五：原生工业场景定性可视化与出版级 4 行 $\times$ 5 列对比图件生成

#### 1. 原生工业场景全量场景 2D/3D 可视化渲染（全量覆盖各工件 scene_list 中的全部场景）
在无真值标注的 ITODD 原生工业场景中，对三大工件的所有原生工业场景（Star 共 50 个场景、Screw_Black 共 23 个场景、Bracket_Planar 共 39 个场景）进行全量批处理匹配与渲染。分别使用官方默认参数（`native_default_<model>`）与 Real Oracle 5000 轮最优参数（`native_<model>_oracle`）渲染保存 2D 轮廓投影叠加图、3D 点云配准图及元数据 JSON：

```powershell
# 1.1 星形调节把手 star (全量 50 个场景批量渲染)
uv run python visualize_results.py --model star --dataset-type native --out-dir visualizations/native_default_star --min-score 0.5
uv run python visualize_results.py --model star --dataset-type native --storage-dir results/itodd_manual_annotated_star_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_star_oracle --min-score 0.5

# 1.2 黑色六角螺栓 screw_black (全量 23 个场景批量渲染)
uv run python visualize_results.py --model screw_black --dataset-type native --out-dir visualizations/native_default_screw_black --min-score 0.01
uv run python visualize_results.py --model screw_black --dataset-type native --storage-dir results/itodd_manual_annotated_screw_black_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_screw_black_oracle --min-score 0.01

# 1.3 薄板工件 bracket_planar (全量 39 个场景批量渲染，开启剥离匹配)
uv run python visualize_results.py --model bracket_planar --dataset-type native --out-dir visualizations/native_default_bracket_planar --min-score 0.5 --peeling
uv run python visualize_results.py --model bracket_planar --dataset-type native --storage-dir results/itodd_manual_annotated_bracket_planar_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_bracket_planar_oracle --min-score 0.5 --peeling
```

#### 2. 出版级 4 行 $\times$ 5 列 2D/3D 定性对比大图一键生成（600 DPI）
自动将上述渲染生成的 4 类图件（第 1 行默认 2D 叠加、第 2 行默认 3D 点云、第 3 行优化 2D 叠加、第 4 行优化 3D 点云）等比例紧密拼装为高分辨率大图，直接输出矢量 `.pdf` 与超清 `.png` 至论文插图目录 `figures/chapter3/`：

```powershell
# 一键生成全部三大典型工件的 4x5 出版级高清对比图 (600 DPI，同时保存至 figures/chapter3/ 与 visualizations/comparison/)
uv run python plotting_scripts/plot_qualitative_comparison.py --all --dpi 600

# 也可针对单一工件单独生成
uv run python plotting_scripts/plot_qualitative_comparison.py --model star --dpi 600
uv run python plotting_scripts/plot_qualitative_comparison.py --model screw_black --dpi 600
uv run python plotting_scripts/plot_qualitative_comparison.py --model bracket_planar --dpi 600
```
> **论文对应成果**：
> - **Figure 3.8 (图件 3_4_star_compare.pdf)**：《Star 星形手柄在原生 ITODD 工业场景下默认基线与 5000 轮 HPO 优化参数的 2D 轮廓投影与 3D 点云匹配定性对比图》；
> - **Figure 3.9 (图件 3_5_screw_compare.pdf)**：《Screw_Black 六角螺栓在原生 ITODD 工业场景下默认基线与 5000 轮 HPO 优化参数的 2D 轮廓投影与 3D 点云匹配定性对比图》；
> - **Figure 3.10 (图件 3_6_bracket_compare.pdf)**：《Bracket_Planar 超薄钣金件在原生 ITODD 工业场景下结合点云递归剥离的 2D 轮廓投影与 3D 点云匹配定性对比图》。

#### 3. 博士论文第三章全量实验图件一键集成生成流水线
```powershell
# 一键自动绘制第三章全部图件 (包含 Figure 3-3 全局收敛曲线、Figure 3-4 目标函数消融、Figure 3-5 剪枝加速分析、Figure 3-6 参数敏感度分析、Figure 3-7 仿真域与真实上限对比，以及上述三大工件 4x5 定性对比大图)
uv run python plotting_scripts/generate_all_figures.py
```

---

### 阶段六：一键汇总全套数据大表
```powershell
uv run python summarize_results.py
```

---

## 5. 常用高频运维、测试与核查命令速查手册 (Quick Reference Manual)

为便于日常开发、环境校验与论文实验复现，以下整理出最常用的单行快捷指令集（均为单行执行，杜绝 PowerShell 换行粘帖错误）：

### 5.0 真实域点云确定性去桌面离线预处理与效果图生成速查 (Deterministic Tabletop Preprocessing & Fig 3.1)
```powershell
# 离线运行 RANSAC 桌面分割 (固定全局随机种子 42 + 物理尺寸距离阈值)，将 19 个真实工业场景纯工件点云固化至 points_tabletop_filtered/*.ply
uv run python preprocess_filter_tabletop.py

# 交互式查看与微调 RANSAC 桌面分割效果（支持按 [C] 一键捕获去白边高清截图）：
uv run python visualize_ransac_tabletop.py --scene-id 8 --model star --thresh 1.0

# 一键生成博士论文图 3.1（确定性 RANSAC 桌面点云分割前后对比图件 3_1_ransac_tabletop_filter.pdf / .png）：
uv run python plotting_scripts/plot_fig1_ransac_tabletop.py
```

### 5.1 架构与功能回归测试 (Testing & Health Check)
```powershell
# 运行完整测试套件（61+ 项单元测试，包括协议、关联度量、采样器与数据加载）
uv run pytest tests/

# 仅测试数据加载与协议对齐
uv run pytest tests/test_dataset_protocol.py tests/test_bop_scene_loader.py

# 仅测试匹配流水线与严格关联度量
uv run pytest tests/test_pipeline.py tests/test_strict_association.py
```

### 5.2 3D 点云 PLY 导出与 CloudCompare 标定核查 (PointCloud PLY Export)
```powershell
# 1. 一键全量导出全部 6 大典型场景的彩色点云（同时导出 BOP 与 Native）
uv run python export_pointcloud_ply.py --dataset-type all --out-dir data/exported_ply

# 2. 导出 BOP itoddmv_val 点云（stride=1 全分辨率）
uv run python export_pointcloud_ply.py --dataset-type bop --bop-images "0,3,293,296,450,468" --stride 1 --out-dir data/exported_ply/bop_full

# 3. 导出 ITODD Native 点云（stride=1 全分辨率）
uv run python export_pointcloud_ply.py --dataset-type native --scenes "0,3,293,296,450,468" --stride 1 --out-dir data/exported_ply/native_full
```

### 5.3 默认基线定量评测速查 (Default Baseline Quantitative Evaluation)
```powershell
# 合成域 Dev 集默认参数评测 (Scene 0 Dev)
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model bracket_planar --split dev --results-root results/baseline_default_scene0/bracket_planar --run-id eval-default-dev
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model screw_black --split dev --results-root results/baseline_default_scene0/screw_black --run-id eval-default-dev
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model star --split dev --results-root results/baseline_default_scene0/star --run-id eval-default-dev

# 真实域手动标注数据集默认参数评测 (自动探查并加载 points_tabletop_filtered/*.ply 固化点云)
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model bracket_planar --split dev --filter-tabletop --results-root results/baseline_default_manual/bracket_planar --run-id eval-default-manual-ransac
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model screw_black --split dev --filter-tabletop --results-root results/baseline_default_manual/screw_black --run-id eval-default-manual-ransac
uv run python main.py evaluate-default --bop-manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --model star --split dev --filter-tabletop --results-root results/baseline_default_manual/star --run-id eval-default-manual-ransac
```

### 5.4 真实域 1000 轮 Real-to-Real Oracle 极限寻优 (Auto RANSAC Tabletop Removal)
```powershell
# 薄板 bracket_planar 1000 轮寻优 (含确定性去桌面与 10 个 Checkpoints 评测)
uv run python run_experiment.py --model bracket_planar --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_bracket_planar_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42

# 黑色螺丝 screw_black 1000 轮寻优
uv run python run_experiment.py --model screw_black --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_screw_black_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42

# 星形调节把手 star 1000 轮寻优
uv run python run_experiment.py --model star --sampler TPE --pruner Median --budget 1000 --manifest data/manifests/itodd_manual_annotated/bop_manifest.csv --results-root results/itodd_manual_annotated_star_tpe_median_lexrecall_b1000_s42 --allow-query-overlap --seed 42
```

### 5.5 论文 4K 超高清定性可视化全量场景批量渲染速查 (Visualizations)
```powershell
# 1. 原生工业场景默认参数全量批量渲染 (自动加载对应模型 scene_list_<model>.txt 的全部场景)
uv run python visualize_results.py --model star --dataset-type native --out-dir visualizations/native_default_star --min-score 0.5
uv run python visualize_results.py --model screw_black --dataset-type native --out-dir visualizations/native_default_screw_black --min-score 0.01
uv run python visualize_results.py --model bracket_planar --dataset-type native --out-dir visualizations/native_default_bracket_planar --min-score 0.5 --peeling

# 2. 原生工业场景加载 Real Oracle 5000 轮最优参数全量批量渲染 (自动从 b5000 数据库读取最优参数)
uv run python visualize_results.py --model star --dataset-type native --storage-dir results/itodd_manual_annotated_star_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_star_oracle --min-score 0.5
uv run python visualize_results.py --model screw_black --dataset-type native --storage-dir results/itodd_manual_annotated_screw_black_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_screw_black_oracle --min-score 0.01
uv run python visualize_results.py --model bracket_planar --dataset-type native --storage-dir results/itodd_manual_annotated_bracket_planar_tpe_median_lexrecall_b5000_s42/studies --out-dir visualizations/native_bracket_planar_oracle --min-score 0.5 --peeling
```

### 5.6 论文定性对比大图与全章图件出版级生成速查 (Publication Figures)
```powershell
# 1. 出版级 4行x5列 2D/3D 定性对比大图一键生成 (600 DPI，生成至 figures/chapter3/ 与 visualizations/comparison/)
# 从项目子目录 code/hpo-3d-match/ 运行:
uv run python plotting_scripts/plot_qualitative_comparison.py --all --dpi 600

# 若从论文根目录 graduate-thesis/ 运行:
uv run python code/hpo-3d-match/plotting_scripts/plot_qualitative_comparison.py --all --dpi 600

# 2. 博士论文第三章全量实验图件一键集成生成流水线 (生成 Figure 3-3 至 3-7 全部插图与对比大图)
# 从项目子目录 code/hpo-3d-match/ 运行:
uv run python plotting_scripts/generate_all_figures.py

# 若从论文根目录 graduate-thesis/ 运行:
uv run python code/hpo-3d-match/plotting_scripts/generate_all_figures.py
```

---

### 真实域手动标注数据集场景全量覆盖明细 (共 19 个真实工业场景):
- **`star`** (7 个场景): 8, 10, 13, 164, 749 (手动标定场景) + 0, 3 (转换对齐场景)
- **`bracket_planar`** (7 个场景): 456, 461, 481, 476, 488 (手动标定场景) + 450, 468 (转换对齐场景)
- **`screw_black`** (5 个场景): 299, 302, 305 (手动标定场景) + 293, 296 (转换对齐场景)