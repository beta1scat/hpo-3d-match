# 3D 点云位姿估计超参数优化实验方案规范 (EXPERIMENT_PLAN.md)

**活动实验协议 ID**：`hpo-3d-match-scene0-matrix-v2`  
**标准配置文件**：`configs/hpo_protocol.json`  
**合成域训练清单**：`data/manifests/itodd_scene0_v1/bop_manifest.csv` (Scene 0 PBR, 8:2 Train/Dev)  
**真实域验证清单**：`data/manifests/itoddmv_val/bop_manifest.csv` (ITODD-MV 真实标注验证集)  
**原生工业场景清单**：`data/manifests/itodd_external/itodd_external_manifest.csv` (data/3d_long_baseline 原生工业点云)  
**统一实验执行引擎**：`run_experiment.py` / `main.py`  
**结果规整与汇总工具**：`summarize_results.py`  
**论文标准 4K 可视化工具**：`visualize_results.py`  
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
2. **迁移验证（Sim-to-Real）**：将合成域选出的最佳模型参数直接迁移至完全独立的真实世界数据集 `itoddmv_val` 上进行定量评测（输出 TP, FP, FN, F1, Recall）。
3. **真实理论上限（Real Oracle）**：将 `itoddmv_val` 接入优化器寻优，确立工业无序抓取的理论物理性能极限。
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

### 阶段四：Sim-to-Real 定量迁移验证（在真实数据集 itoddmv_val 上评估）
将合成域优化出的最佳参数模型直接加载并在真实世界场景 `itoddmv_val` 上进行定量评测：
```powershell
# 薄板工件 Sim-to-Real 评测
uv run python main.py evaluate-best --bop-manifest data/manifests/itoddmv_val/bop_manifest.csv --study-bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model bracket_planar --split test --storage-dir results/bracket_planar_tpe_nop_lexrecall_b500_s42/studies --sampler TPE --pruner Nop --repeat 0 --seed 42 --results-root results/bracket_planar_tpe_nop_lexrecall_b500_s42/evaluations --run-id eval-sim2real-val

# 黑色螺丝 Sim-to-Real 评测
uv run python main.py evaluate-best --bop-manifest data/manifests/itoddmv_val/bop_manifest.csv --study-bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model screw_black --split test --storage-dir results/screw_black_tpe_nop_lexrecall_b500_s42/studies --sampler TPE --pruner Nop --repeat 0 --seed 42 --results-root results/screw_black_tpe_nop_lexrecall_b500_s42/evaluations --run-id eval-sim2real-val

# 星形手柄 Sim-to-Real 评测
uv run python main.py evaluate-best --bop-manifest data/manifests/itoddmv_val/bop_manifest.csv --study-bop-manifest data/manifests/itodd_scene0_v1/bop_manifest.csv --model star --split test --storage-dir results/star_tpe_nop_lexrecall_b500_s42/studies --sampler TPE --pruner Nop --repeat 0 --seed 42 --results-root results/star_tpe_nop_lexrecall_b500_s42/evaluations --run-id eval-sim2real-val
```

---

### 阶段五：真实域理论上限 (Oracle) 与原生场景 4K 超高清定性可视化

#### 1. Real Oracle 寻优（在 itoddmv_val 上运行 500 轮获得物理极限参数）
```powershell
uv run python run_experiment.py --model bracket_planar --sampler TPE --pruner Median --budget 500 --manifest data/manifests/itoddmv_val/bop_manifest.csv --seed 42
uv run python run_experiment.py --model screw_black --sampler TPE --pruner Median --budget 500 --manifest data/manifests/itoddmv_val/bop_manifest.csv --seed 42
uv run python run_experiment.py --model star --sampler TPE --pruner Median --budget 500 --manifest data/manifests/itoddmv_val/bop_manifest.csv --seed 42
```
> **论文对应成果**：**Table 3.7**：《Sim-to-Real 迁移泛化性与 Real-to-Real 理论天花板性能对比表》。

#### 2. 原生工业场景 4K 满幅定性可视化（ITODD data/3d_long_baseline）
自动读取优化参数，加载原生传感器 X/Y/Z 空间深度与 2D 灰度图，输出 4K 超高清 2D 投影图、3D 真实纹理点云匹配图及元数据 JSON：
```powershell
# 薄板 bracket_planar 在 ITODD 原生 0-4 场景中进行 4K 满幅渲染
uv run python visualize_results.py --model bracket_planar --dataset-type native --storage-dir results/bracket_planar_tpe_nop_lexrecall_b500_s42/studies --scenes "0,1,2,3,4" --out-dir visualizations/native_bracket_planar

# 黑色螺丝 screw_black 在 ITODD 原生 0-4 场景中进行 4K 满幅渲染
uv run python visualize_results.py --model screw_black --dataset-type native --storage-dir results/screw_black_tpe_nop_lexrecall_b500_s42/studies --scenes "0,1,2,3,4" --out-dir visualizations/native_screw_black

# 星形手柄 star 在 ITODD 原生 0-4 场景中进行 4K 满幅渲染
uv run python visualize_results.py --model star --dataset-type native --storage-dir results/star_tpe_nop_lexrecall_b500_s42/studies --scenes "0,1,2,3,4" --out-dir visualizations/native_star
```
> **论文对应成果**：**Figure 3.5**：《三大典型工件在原生 ITODD 工业场景下的 3D 点云匹配与 2D 轮廓投影定性验证图》。

---

### 阶段六：一键汇总全套数据大表
```powershell
uv run python summarize_results.py
```
