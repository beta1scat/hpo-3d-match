# 原生 ITODD 场景 6D 位姿交互式手动标注与 BOP 数据集导出使用指南

本文档为工件 6D 位姿交互式手动标注工具 [`manual_annotate_bop.py`](manual_annotate_bop.py) 的完整技术说明书，旨在指导研究人员完成原生 ITODD 场景的点云位姿真值标注，并自动化生成符合 BOP（Benchmark for 6D Object Pose Estimation）官方规范的数据集。

---

## 1. 系统设计与坐标系规范

### 1.1 物理尺度与坐标系统一
- **单位统一为毫米 ($\text{mm}$)**：
  - CAD 模型（`obj_000005.ply`, `obj_000024.ply`, `obj_000025.ply`）几何顶点单位为毫米。
  - 原生相机点云传感器数据（$X, Y, Z$ 单位为米）在加载时统一乘以 $1000.0$ 转换为毫米。
  - 交互过程中记录的平移向量 $t = [t_x, t_y, t_z]^\top$ 直接以毫米为单位输出，旋转矩阵 $R \in SO(3)$ 保持正交无量纲。写入 BOP 格式时零跨单位转换，消除精度损失。
- **料框中心智能初始化**：
  - 活动模型加载时默认自动锚定于 ITODD 原生料框几何中心 $t_0 \approx [3.06, -0.23, 721.29]\,\text{mm}$，避免因模型处于原点 $(0,0,0)$ 引起视角丢失。
- **高对比度灰度纹理渲染**：
  - 自动抽取原生传感器灰度强度图 `3d_long_baseline_l.tif`，并归一化映射至三维点云的 RGB 属性，使料框阴影与工件反光边缘清晰可辨。

---

## 2. 交互按键操作手册 (Keybindings)

标注视口启动后，Open3D 交互窗口将捕获键盘事件。详细键位映射如下表所示：

| 键位分类 | 键位 (Key) | 功能描述 | 详细操作逻辑 |
| :--- | :--- | :--- | :--- |
| **流程控制** | **`S`** | **保存真值位姿** | 锁定当前黄色模型位姿，视口中生成**翠绿色**确认模型；重置活动模型以继续标注下一个实例 |
| | **`N`** | **下一场景并导出 BOP** | 自动完成当前场景深度图转换、相机内参与真值 JSON 写入、清单维护，随后加载下一场景 |
| | **`Z`** | **撤销 (Undo)** | 撤销并从视口与记录中移除当前场景最近保存的一个真值实例 |
| | **`R`** | **重置位姿 (Reset)** | 将当前活动模型重置回初始料框中心 |
| | **`ESC`** | **安全退出** | 安全终止当前标注会话，已保存并导出的场景数据完整保留 |
| | **`H`** | **控制台帮助** | 在终端打印键位对照表及当前场景实例统计 |
| **高精对齐** | **`F` / `Space`** | **ICP 自动吸附** | 在工件 35mm 局部邻域内执行点对面（Point-to-Plane）ICP，一键亚毫米精准贴合 |
| **步长切换** | **`1`** | **微调档 (Fine)** | 平移步长 $\Delta t = 0.5\,\text{mm}$，旋转步长 $\Delta \theta = 1.0^\circ$ |
| | **`2`** | **标准档 (Medium)** | 平移步长 $\Delta t = 2.5\,\text{mm}$，旋转步长 $\Delta \theta = 5.0^\circ$（默认档位） |
| | **`3`** | **粗调档 (Coarse)** | 平移步长 $\Delta t = 10.0\,\text{mm}$，旋转步长 $\Delta \theta = 15.0^\circ$ |
| **平移控制** | **`A` / `D`** | **X 轴平移** | `A`: 沿相机 X 轴负向向左；`D`: 沿相机 X 轴正向向右 |
| | **`W` / `X`** | **Y 轴平移** | `W`: 沿相机 Y 轴负向上移；`X`: 沿相机 Y 轴正向下移 |
| | **`Q` / `E`** | **Z 轴平移** | `Q`: 沿相机光轴靠近传感器；`E`: 沿相机光轴远离传感器 |
| **旋转控制** | **`I` / `K`** | **俯仰角 (Pitch)** | `I`: 绕相机 X 轴正转；`K`: 绕相机 X 轴反转 |
| | **`J` / `L`** | **偏航角 (Yaw)** | `J`: 绕相机 Y 轴反转；`L`: 绕相机 Y 轴正转 |
| | **`U` / `O`** | **翻滚角 (Roll)** | `U`: 绕相机光轴自旋反转；`O`: 绕相机光轴自旋正转 |

---

## 3. 标准推荐标注流程 (Standard Operating Procedure)

1. **场景加载**：执行启动命令后，Open3D 视口显示场景灰度点云与明黄色活动 CAD 模型。
2. **粗调定位**：
   - 按 `3` 键切换到粗调档，通过 `A/D/W/X/Q/E` 平移模型至目标零件大致位置；
   - 通过 `I/K/J/L/U/O` 旋转模型至目标零件的大致朝向。
3. **精细吸附 (推荐)**：
   - 按 `2` 键切换至标准档微调，使模型与场景点云轮廓基本吻合（轮廓误差 $< 5\sim 10\,\text{mm}$）；
   - 按下 **`F`** 或 **`空格键 (Space)`** 触发局部点对面 ICP，算法将自动贴合表面，终端输出位移残差及 RMSE；
   - 若贴合满意，直接进入下一步；若有微小偏差，可按 `1` 键切换到微调档手动补偿。
4. **保存当前实例**：
   - 按下 **`S`** 键，当前黄色模型转变为**翠绿色**并固定；
   - 终端打印保存日志，活动黄色模型自动重置并略微偏移，等待标注场景中的下一个同类零件。
5. **保存场景并切入下一场景**：
   - 当前场景内所有目标工件标注完毕后，按下 **`N`** 键；
   - 系统自动生成 BOP 深度图、写入真值元数据并追加清单，随后自动打开下一个场景。

---

## 4. BOP 导出文件结构规范

所有导出的数据集按照 BOP 标准结构沉淀至独立目录：

```text
data/
├── itodd_manual_annotated/
│   └── val/
│       └── 000001/
│           ├── depth_3dlong/
│           │   ├── 000008.tif          # 单通道 float32 深度图，单位毫米
│           │   ├── 000010.tif
│           │   └── ...
│           ├── scene_camera_3dlong.json # 相机内参 cam_K 与外参
│           ├── scene_gt_3dlong.json     # 6D 位姿真值 (cam_R_m2c, cam_t_m2c, obj_id)
│           └── scene_gt_info_3dlong.json# 可视度与包围盒元数据 (visib_fract, bbox)
└── manifests/
    └── itodd_manual_annotated/
        └── bop_manifest.csv            # Optuna 寻优与评估流程专用数据清单
```

---

## 5. 快速启动命令

### 5.1 单工件模块启动命令
```powershell
# 标注 star 星形工件（场景: 8, 10, 13, 164, 749）
uv run python manual_annotate_bop.py --model star

# 标注 bracket_planar 平板支架（场景: 456, 461, 481, 476, 488）
uv run python manual_annotate_bop.py --model bracket_planar

# 标注 screw_black 黑色螺栓（场景: 299, 302, 305）
uv run python manual_annotate_bop.py --model screw_black
```

### 5.2 全量连续启动命令
```powershell
# 按顺序连续标注全部 3 类工件共 13 个场景
uv run python manual_annotate_bop.py --all
```

---

## 6. 3D 场景与模型对齐效果可视化检查工具

研究人员可随时使用 [`visualize_annotation.py`](visualize_annotation.py) 脚本独立查看点云场景与 CAD 模型的叠加对齐状态（支持在标注前检查初始相对尺度与料框位置，或在标注后检查已保存的真值贴合质量）：

### 6.1 核心特性
- **连续多场景遍历**：支持按类别（`--model star`）、指定序列（`--scenes 8,10,13`）或全量 19 个场景（`--all`）连续浏览。
- **极简切换交互**：视口内直接按下 **`N`** 键或 **`空格键 (Space)`** 即可快速加载下一张场景，按 **`ESC`** 键随时安全退出。
- **真值自动变换叠加**：自动读取 `scene_gt_3dlong.json`，将对应 CAD 模型应用 $(R, t)$ 矩阵放置于真实空间，并以翠绿、青蓝、橙色等多实例区分色彩高亮渲染。
- **辅助定位标尺**：附加局部 3D 坐标轴（RGB: X红/Y绿/Z蓝）直观提示相机坐标系原点与方向。

### 6.2 连续检查运行命令
```powershell
# 1. 依次连续检查 star 模型的全部 7 个场景（0, 3, 8, 10, 13, 164, 749）
uv run python visualize_annotation.py --model star

# 2. 依次连续检查 bracket_planar 模型的全部 7 个场景（450, 456, 461, 468, 476, 481, 488）
uv run python visualize_annotation.py --model bracket_planar

# 3. 依次连续检查 screw_black 模型的全部 5 个场景（293, 296, 299, 302, 305）
uv run python visualize_annotation.py --model screw_black

# 4. 全量贯穿检查全部 19 个标注场景
uv run python visualize_annotation.py --all

# 5. 检查特定单个场景
uv run python visualize_annotation.py --scene-id 8 --model star
```

---

## 7. 导出手工标注点云与真值 CAD 至 PLY (支持 CloudCompare 检验)

研究人员可使用更新后的 [`export_pointcloud_ply.py`](export_pointcloud_ply.py) 脚本将手工标注的场景深度图及变换后的真值 CAD 模型全量导出为 PLY 文件，以便直接在 CloudCompare 或 MeshLab 等第三方工具中进行多视图复查与距离测量：

```powershell
# 1. 仅导出手工标注的场景点云及对应真值 CAD 模型（默认米单位，输出至 data/exported_ply/manual_annotated）
uv run python export_pointcloud_ply.py --dataset-type manual

# 2. 以毫米 (mm) 单位导出，匹配 CAD 模型原生尺度
uv run python export_pointcloud_ply.py --dataset-type manual --scale mm

# 3. 指定导出手工标注场景 8 的点云
uv run python export_pointcloud_ply.py --dataset-type manual --manual-images 8 --scale mm
```

---

## 8. 官方 itoddmv_val 数据集偏差弥补与格式统一转换

为了将官方原有的 6 个评测场景（`0, 3, 293, 296, 450, 468`）与新手工标注的 13 个场景合并，构建一个物理坐标系统一、无任何高度偏差（弥补约 $3.41\,\text{mm}$ 的虚拟相机外参误差）的全量训练/验证集，研究人员可运行转换脚本 [`convert_itoddval_to_manual_format.py`](convert_itoddval_to_manual_format.py)：

### 8.1 核心转换机制
1. **深度重构**：抛弃官方 BOP 虚拟相机重投影深度，直接从物理原生传感器生成 float32 毫米深度图；
2. **真值位姿外参逆变换补偿**：计算 BOP 虚拟坐标系到物理传感器原生坐标系的刚体变换 $T_{\text{bop}\to\text{nat}}$，自动补偿真值位姿：
   \[
   T_{\text{m2nat}} = T_{\text{bop}\to\text{nat}} \cdot T_{\text{m2bop}} \quad (\Delta Z \approx -3.41\,\text{mm})
   \]
3. **清单无缝追加**：自动将转换后的场景合并注入 [`data/manifests/itodd_manual_annotated/bop_manifest.csv`](data/manifests/itodd_manual_annotated/bop_manifest.csv)。

### 8.2 运行转换命令
```powershell
# 一键转换全量 6 个场景并弥补偏差
uv run python convert_itoddval_to_manual_format.py

# 单独转换并检查特定场景（例如场景 0）
uv run python convert_itoddval_to_manual_format.py --scenes 0
```

---

## 9. 动态 RANSAC 桌面点云拟合与交互式剔除工具

为了验证通过算法动态剔除料框底面/桌面点云的可行性，研究人员可运行交互式工具 [`visualize_ransac_tabletop.py`](visualize_ransac_tabletop.py)。该工具支持在三维视口中以毫米级步长动态调整 RANSAC 平面拟合距离阈值，并实时渲染剔除效果：

### 9.1 核心按键与动态调节
- **阈值微调**：按 **`方向键上 (Up)`** 或 **`+`** 增大阈值（$+0.2\,\text{mm}$）；按 **`方向键下 (Down)`** 或 **`-`** 减小阈值（$-0.2\,\text{mm}$）；
- **阈值粗调**：按 **`方向键右 (Right)`** 或 **`]`** 增加 $+1.0\,\text{mm}$；按 **`方向键左 (Left)`** 或 **`[`** 减小 $-1.0\,\text{mm}$；按数字键 **`1~9`** 直达 $1.0\sim 9.0\,\text{mm}$；
- **桌面显隐模式 (`T` 键)**：
  - **模式 1 (红色高亮)**：桌面内点高亮显示为鲜红色，工件保留真实纹理；
  - **模式 2 (完全剔除)**：桌面点云完全隐藏，仅保留悬浮在料框中的工件，直观观察工件底面是否被误切；
  - **模式 3 (深灰底色)**：桌面点云呈现微弱深灰色，突显高亮工件。
- **真值 CAD 辅助比对 (`G` 键)**：切换显示 Ground Truth CAD 模型，直观检查 RANSAC 拟合平面与工件 CAD 底部的实际几何间隙；
- **保存干净工件点云 (`S` 键)**：一键将剔除桌面后的工件点云导出至 `data/exported_ply/ransac_filtered/`。

### 9.2 运行命令
```powershell
# 1. 启动 star 场景 8 的动态 RANSAC 桌面剔除（默认标定阈值 1.0mm）
uv run python visualize_ransac_tabletop.py --scene-id 8 --model star --thresh 1.0

# 2. 连续遍历检查 bracket_planar 零件各场景（标定阈值 0.8mm）
uv run python visualize_ransac_tabletop.py --model bracket_planar --thresh 0.8

# 3. 连续遍历检查 screw_black 零件各场景（标定阈值 4.0mm）
uv run python visualize_ransac_tabletop.py --model screw_black --thresh 4.0
```




