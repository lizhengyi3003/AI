# Copilot 提交信息生成规范 — ResNeXt 图像分类项目

> **版本**: v1.1.0 | **日期**: 2026-05-22 | **项目**: 基于轻量化 ResNeXt 的图像分类

---

本文件作为 Copilot 生成 Git 提交信息的指令，所有生成的提交信息必须严格遵循以下规范。
以下规范专为 **ResNeXt 图像分类训练项目** 定制。

---

## 零、文件来源约束

body 中列出的每一个文件路径，**必须来自上下文中实际存在的 diff 内容**，禁止：

- 编造任何未出现在 diff 中的文件路径
- 推测或描述 diff 中未体现的功能变更
- 将多个文件的变更合并为一条（除非规则四第 5、6 条明确允许）

---

## 一、整体结构

```
type(作用域): 简短中文主题

### 模块名
- `相对文件路径`：做了什么 → 产生的影响

BREAKING CHANGE: 描述破坏性变更（仅在有不向下兼容的变更时写）
Closes #N（仅在关联 issue 时写）
```

- **首行**（header）：不超过 70 个字符，不加句号
- **空行**：header 与 body 之间必须有一个空行；body 与 footer 之间必须有一个空行
- **body**：按模块目录分组，每个文件单独一条，使用反引号标注文件路径
- **footer**：仅在有破坏性变更或关联 issue 时写，否则省略

---

## 二、提交类型（type）

| 类型       | 用途                                                         |
| -------- | ---------------------------------------------------------- |
| feat     | 新增功能（如新增数据增强方法、新增模型组件、新增推理模式）                       |
| fix      | 修复 Bug（如修复指标计算错误、修复数据加载异常、修复设备兼容性问题）                 |
| refactor | 代码重构（如重组项目目录结构、重命名模块、简化代码逻辑）                          |
| perf     | 性能优化（如优化 DataLoader 参数、减少显存占用、加速推理）                     |
| style    | 代码格式调整（不影响逻辑，如注释补全、空格缩进、命名规范）                          |
| test     | 新增或修改测试代码（`test.py` 或测试相关脚本）                              |
| docs     | 文档变更（仅限 `help_document/`、`README.md` 等说明文件）                  |
| build    | 构建系统或外部依赖变更（如 `requirements.txt`、`.gitignore`、DVC 配置）        |
| chore    | 杂项变更（如 `.dvc/config` 远程存储配置、`.gitignore` 规则、VS Code 工作区配置）  |
| revert   | 撤销某次提交                                                     |
| ci       | CI/CD 流程变更                                                 |

**type 选型优先级**：当一次提交同时涉及多种类型时，按以下优先级选择主 type：

```
fix > feat > refactor > perf > docs > build > chore
```

---

## 三、作用域（scope）词汇表

scope 使用**中文**，从以下列表中选择最贴近的一个；如变更跨越两个及以上模块，用 `&` 连接所有涉及的模块名（如 `模型&训练`），**禁止**使用笼统的「多模块」标签：

| scope  | 对应文件 / 目录                              |
| ------ | --------------------------------------- |
| 模型     | `model.py` — ResNeXt 网络结构定义            |
| 数据     | `mydataset.py`、`data/`、`data.dvc`        |
| 训练     | `train.py` — 训练主脚本                      |
| 测试     | `test.py` — 测试评估脚本                      |
| 环境     | `environment/`（`device_config.py`、`device_utils.py`、`install_pytorch.py`、`verify_setup.py`） |
| 工具     | `utils/`（`utils.py`、`predict.py`、`report.py`、`draw.py`） |
| 构建     | `requirements.txt`、`.gitignore`、`AI.code-workspace` |
| 文档     | `help_document/`、`README.md`、`EXPERIMENT_LOG.md`、`PATH_STRUCTURE_UPDATE.md` |
| 日志     | `log/{exp_id}/`（`training/train_log.txt`、`evaluation/`）、可视化图表（`.png`） |
| 模型输出   | `model-out/{exp_id}/`（`best.pth`、`last.pth`）、`model-out.dvc` |
| DVC    | `.dvc/`、`*.dvc` — DVC 数据/模型版本追踪       |

**多模块 scope 示例**：`fix(模型&训练)`、`refactor(环境&工具&构建)`、`docs(文档&数据)`

---

## 四、body 书写规则

1. 按**模块目录**分组，组标题格式为 `### 模块名`（与 scope 词汇表中的中文名对应）
2. 每个文件一条 bullet：`` - `路径/文件名`：[做了什么] → [影响是什么] ``
3. "做了什么"使用具体动词描述操作；"影响是什么"描述对功能、性能、接口或数据的影响
4. 若同模块内只有一个文件变动，组标题可省略，直接写 bullet
5. 纯格式调整（空行、注释措辞、缩进）等琐碎改动可合并为一条，注明"统一格式"
6. 多个同类文件的批量操作（如 `environment/` 下 4 个文件统一移入子目录）可合并为一条，注明文件数量

---

## 五、特殊文件处理规范

### DVC 追踪文件（`.dvc`）

- 归入 `DVC` 或对应数据/模型分组
- diff 中仅显示 hash/size 变化时，描述格式为：
  ```
  - `data.dvc`：DVC 追踪更新 → 同步最新数据集版本
  - `model-out.dvc`：DVC 追踪更新 → 同步最新模型权重版本
  ```
- 若首次新增 `.dvc` 文件，描述为"新增 DVC 追踪文件 → 纳入版本管理"

### DVC 配置文件（`.dvc/config`）

- 归入 `构建` 分组
- **禁止**在提交信息中写入实际的 access_key、secret_key 等敏感凭证值
- 描述格式：
  ```
  - `.dvc/config`：新增远程存储配置 → 支持 DVC 数据/模型远端同步
  ```

### 模型权重文件（`.pth`）

- `model-out/*.pth` 文件通常由 `.gitignore` 排除，仅通过 DVC 追踪
- 若 diff 中出现，说明为首次添加或 `.gitignore` 规则变更，描述权重文件用途（如"最佳验证准确率权重"、"最后一个 epoch 权重"）
- 不描述文件大小或具体数值

### 训练日志文件（`train_log.txt`）

- 归入 `日志` 分组
- 路径格式为 `log/{exp_id}/training/train_log.txt`，exp_id 为实验编号（如 01、02）
- 首次新增时描述为"新增实验{exp_id}训练日志，记录 N 个 epoch 的 loss 和 accuracy"
- 后续更新时描述为"实验{exp_id}：追加第 N~M 个 epoch 的训练记录"

### 数据集目录（`data/`）

- `data/` 目录通常由 `.gitignore` 排除，仅通过 DVC 追踪
- 若变更涉及数据集路径调整或数据加载逻辑，归入 `数据` 分组

---

## 六、footer 书写规则

- **破坏性变更**：`BREAKING CHANGE: <描述接口/行为变化，以及迁移方式>`
- **关联 issue**：`Closes #<编号>` 或 `Refs #<编号>`（多个用换行分隔）
- 无破坏性变更且无关联 issue 时，**不写 footer**

---

## 七、完整示例（基于本项目）

### 示例 1 — 修复训练指标计算 Bug

```
fix(训练): 修复训练指标计算使用错误分母导致的类型错误

### 训练
- `train.py`：将准确率计算中的 `len(loader.dataset)` 替换为 `train_total`/`val_total` 累积计数器 → 消除类型不匹配错误，指标统计更可靠
```

---

### 示例 2 — 重构项目目录结构

```
refactor(环境&工具&训练): 将环境配置与工具脚本移入子目录，统一项目结构

### 环境
- `device_config.py`、`device_utils.py`、`install_pytorch.py`、`verify_setup.py`（4个文件）：移入 `environment/` 子目录 → 环境相关脚本集中管理

### 工具
- `predict.py`、`utils.py`（2个文件）：移入 `utils/` 子目录 → 工具脚本集中管理

### 训练
- `train.py`：更新 import 路径至 `environment.device_utils` → 适配目录重组
- `test.py`：更新 import 路径至 `environment.device_utils` → 适配目录重组
```

---

### 示例 3 — 新增模型推理功能

```
feat(工具): 新增推理预测脚本，支持单张/批量/交互式三种模式

### 工具
- `utils/predict.py`：新增 `predict_single()` 单张预测、`predict_batch()` 批量预测、交互式命令行三种推理模式，补充完整 Google 风格 docstring → 提供开箱即用的模型推理能力

### 文档
- `README.md`：新增"模型推理与预测"章节，包含命令行用法和 Python API 示例 → 用户可直接参考文档进行推理
```

---

### 示例 4 — 含 DVC 和文档的多模块提交

```
docs(文档&数据): 完善 README 文档并新增 DVC 数据追踪

### 文档
- `README.md`：新增实验结果与日志章节，补全常见问题与解决方案，更新文件结构树，添加 `environment/` 和 `utils/` 目录说明 → 文档覆盖训练全流程
- `help_document/copilot-commit-instructions.md`：重写 scope 词汇表和特殊文件处理规范 → 适配当前 AI 训练项目

### DVC
- `data.dvc`：新增 DVC 追踪文件 → 纳入数据集版本管理
- `model-out.dvc`：新增 DVC 追踪文件 → 纳入模型权重版本管理

### 构建
- `.dvc/config`：新增远程存储配置 → 支持 DVC 远端同步
```

---

### 示例 5 — 模型结构变更 + 训练参数调整

```
feat(模型&训练): ResNeXt 新增 Dropout 正则化，训练启用混合精度

### 模型
- `model.py`：在 `ResNeXtBlock` 的瓶颈结构后新增 `nn.Dropout2d(p=0.1)` → 抑制过拟合，验证准确率提升约 2%

### 训练
- `train.py`：新增 `--amp` 命令行参数，使用 `torch.cuda.amp` 自动混合精度训练 → GPU 显存占用降低约 30%，训练速度提升约 1.5 倍

### 构建
- `requirements.txt`：新增 `torch>=1.10.0` 最低版本约束 → 确保混合精度 API 可用
```

---

## 八、注意事项

- **禁止使用** `update`、`modify`、`change` 等模糊动词，应使用具体动词：新增、修复、移除、重构、优化、迁移、统一、解耦、重写、补全、简化、调整等
- **header 禁止**以句号结尾
- body 中的文件路径使用**项目根目录为基准的相对路径**
- 单文件提交：body 只写一条 bullet，无需分组
- 提交信息全程使用**中文**（type、scope 中的英文关键字除外）
- **不推测**：若 diff 中某文件改动原因不明，如实描述 diff 内容，不编造业务理由
- 涉及 `data/` 和 `model-out/` 的变更通常通过 DVC 管理，Git 提交中仅体现 `.dvc` 追踪文件和 `.gitignore` 规则变更

---

## 版本历史

### v1.0.0 (2026-05-15)

**针对 ResNeXt 图像分类项目的首次定制**

- ✅ scope 词汇表重写：移除 `界面`/`服务层`/`核心逻辑` 等不适用项，新增 `模型`/`训练`/`环境`/`日志`/`模型输出`/`DVC`
- ✅ 特殊文件处理重写：移除 `.db` 数据库文件规范，新增 `.dvc`/`.pth`/`train_log.txt`/`data/` 处理规范
- ✅ 全部示例替换为 AI 训练项目实际场景
- ✅ 版本号重置为 v1.0.0，标注项目名称

**历史版本**:

### v1.1.0 (2026-05-22)

**适配路径结构重构与 `--exp-id` 实验编号系统**

- ✅ scope 词汇表更新：`工具` 新增 `report.py`/`draw.py`；`日志` 改为 `log/{exp_id}/` 子目录；`模型输出` 改为 `model-out/{exp_id}/` 子目录；`文档` 新增 `EXPERIMENT_LOG.md`/`PATH_STRUCTURE_UPDATE.md`
- ✅ 训练日志处理规范更新：路径格式改为 `log/{exp_id}/training/train_log.txt`
- ✅ 版本号升级至 v1.1.0，日期更新为 2026-05-22

- v0.2.0.8: 通用软件项目版本（已废弃，不适用于本项目）
