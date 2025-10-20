<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**项目简介**
---
本项目基于 PyTorch 和 PySide6 开发了一个房价预测系统，使用线性回归模型来预测房屋价格。通过图形用户界面，用户可以方便地输入房屋特征并获取预测结果。

**数据描述**
---
[“House Prices – Advanced Regression
Techniques”](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) 是一个**回归任务**的数据集，目标是
预测住宅的**销售价格（SalePrice）**。其主要用途是练习特征工程（feature engineering）、回归模型（regression
models）、以及数据预处理流程（包括缺失值处理、类别变量编码、连续变量变换等）。

数据源来自美国艾姆斯市（Ames, Iowa）的一批房屋交易数据，包含大量住宅属性特征（如地块面积、房屋建造年份、房间数、位置类型、质量评级、车库、地下室、露台等）。

**数据结构**
---

+ **训练集**（train.csv）包含每幢房屋的 Id、大约 79 个属性特征（不同类型：数值型、类别型、顺序型）以及目标变量 **SalePrice**。
+ **测试集**（test.csv）具有与训练集同样的特征（不含 SalePrice），模型在测试集上的预测结果用于提交评估。
+ 特征大致可分为：
    + 房屋物理属性：例如 LotArea（地块面积）、1stFlrSF / 2ndFlrSF（第一、第二层地上面积）、GrLivArea（地上居住面积）、TotalBsmtSF（地下室总面积）
    + 建造与时间属性：YearBuilt（建造年份）、YearRemodAdd（改造年份）、GarageYrBlt（车库建造年份）、YrSold（售出年份）
    + 房屋构造与状态属性：例如 OverallQual（整体质量评级）、OverallCond（整体状况评级）、ExterQual（外部质量）、BsmtQual（地下室质量）
    + 附加设施或状态属性：例如 GarageType（车库类型）、PoolArea（泳池面积）、Fence（围栏状况）、MiscFeature（杂项设施）
    + 位置/环境属性：Neighborhood（街区）、MSZoning（分区）、LotShape（土地图形）等
+ 类别型特征 & 顺序型特征：例如 “Ex”/“Gd”/“TA”/“Fa”/“Po” 用于质量评级，具有明确顺序关系。

**隐私声明**
---
本应用程序旨在处理您提供的数据以生成定制化的建议和结果。您的隐私至关重要。

**我们不会收集、存储或传输您的个人信息或数据。** 所有处理都在您的设备本地进行（在浏览器或运行时环境中），*
*数据永远不会发送到任何外部服务器或第三方。**

- **本地处理：** 您的数据永远不会离开您的设备。整个分析和生成过程都在本地进行。
- **无数据保留：** 由于没有数据传输，因此不会在任何服务器上存储数据。关闭应用程序通常会清除任何临时本地数据。
- **透明度：** 整个代码库都是开源的。我们鼓励您随时审查[代码](./)以验证您的数据处理方式。

总之，您始终完全控制和拥有自己的数据。

**许可声明**
---
本项目是开源的，可在 [BSD-3-Clause 许可证](LICENCE) 下使用。

简单来说，这是一个非常宽松的许可证，允许您几乎出于任何目的自由使用此代码，包括在专有项目中，只要您包含原始的版权和许可证声明。

欢迎随意分叉、修改并在此作品基础上进行构建！我们只要求您在适当的地方给予认可。

**环境设置**
---
本项目使用 **Python 3.12** 和 [uv](https://docs.astral.sh/uv/) 进行快速的依赖管理和虚拟环境处理。所需的 Python
版本会自动从 [.python-version](.python-version) 文件中检测到。

1. **安装 uv**：  
   如果您还没有安装 `uv`，可以使用以下命令安装：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 此安装方法适用于 macOS 和 Linux。
    ```
   或者，您可以运行以下 PowerShell 命令来安装：
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # 此安装方法适用于 Windows。
    ```

   **💡 推荐**：为了获得最佳体验，请将 `uv` 作为独立工具安装。避免在 `pip` 或 `conda` 环境中安装，以防止潜在的依赖冲突。

2. **添加依赖**：

- 添加主要（生产）依赖：
    ```bash
    uv add <package_name>
    # 这会自动更新 pyproject.toml 并安装包
    ```
- 添加开发依赖：
    ```bash
    uv add <package_name> --group dev
    # 示例：uv add ruff --group dev
    # 这会自动将包添加到 [project.optional-dependencies.dev] 部分
    ```
- 添加其他类型的可选依赖（例如测试、文档）：
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- 从 `requirements.txt` 文件导入依赖：
    ```bash
    uv add -r requirements.txt
    # 这会从 requirements.txt 读取包并将其添加到 pyproject.toml
    ```

3. **移除依赖**

- 移除主要（生产）依赖：
    ```bash
    uv remove <package_name>
    # 这会自动更新 pyproject.toml 并移除包
    ```
- 移除开发依赖：
    ```bash
    uv remove <package_name> --group dev
    # 示例：uv remove ruff --group dev
    # 这会从 [project.optional-dependencies.dev] 部分移除包
    ```
- 移除其他类型的可选依赖：
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **管理环境**

- 使用添加/移除命令后，同步环境：
    ```bash
    uv sync
    ```

**更新日志**
---
本项目使用 [git-changelog](https://github.com/pawamoy/git-changelog)
基于 [Conventional Commits](https://www.conventionalcommits.org/) 自动生成和维护更新日志。

1. **安装**
   ```bash
   pip install git-changelog
   # 或使用 uv 将其添加为开发依赖
   uv add git-changelog --group dev
   ```
2. **验证安装**
   ```bash
   pip show git-changelog
   # 或专门检查版本
   pip show git-changelog | grep Version
   ```
3. **配置**
   确保您在项目根目录有一个正确配置的 `pyproject.toml` 文件。配置应将 Conventional Commits 指定为更新日志样式。以下是示例配置：
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **生成更新日志**
   ```bash
   git-changelog --output CHANGELOG.md
   # 或者使用 uv 运行
   uv run git-changelog --output CHANGELOG.md
   ```
   此命令会根据您的 git 历史记录创建或更新 `CHANGELOG.md` 文件。
5. **推送更改**
   ```bash
   git push origin main
   ```
   或者，使用您 IDE 的 Git 界面（例如，在许多编辑器中的 `Git → Push`）。
6. **注意**：

- 更新日志是根据您的提交消息遵循 Conventional Commits 规范自动生成的。
- 每当您想要更新更新日志时（通常在发布之前或进行重大更改之后），运行生成命令。
