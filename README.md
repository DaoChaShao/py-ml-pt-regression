<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This project develops a house price prediction system using PyTorch and PySide6, implementing a linear regression model
to predict housing prices. The graphical user interface allows users to easily input house features and obtain
prediction results.

**DATA DESCRIPTION**
---
The [“House Prices – Advanced Regression Techniques”](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
dataset is designed for a regression task: **predicting the sale price (SalePrice)** of residential homes. Its primary
purpose is for practicing feature engineering, regression modeling, and data preprocessing workflows (including missing
value handling, categorical encoding, continuous variable transformation, etc).

The dataset originates from a collection of home sale transactions in Ames, Iowa, USA, and includes many property
attributes (such as lot size, house age, number of rooms, neighborhood, quality ratings, garage, basement, porch, etc).

**DATA STRUCTURE**
---

+ The **training set (train.csv)** includes for each home: an Id, about 79 attribute features (various types: numeric,
  categorical, ordinal) and the target variable **SalePrice**.
+ The **test set (test.csv)** contains the same features (except SalePrice) and is used for making predictions for
  submission/evaluation.
+ The features can be broadly grouped into:
    + Physical attributes: e.g., LotArea (lot size), 1stFlrSF / 2ndFlrSF (first/second floor area), GrLivArea (
      above-ground living area), TotalBsmtSF (basement total area)
    + Construction/time attributes: YearBuilt (year built), YearRemodAdd (year of remodel), GarageYrBlt (garage built
      year), YrSold (year sold)
    + Structural/quality attributes: e.g., OverallQual (overall quality rating), OverallCond (overall condition rating),
      ExterQual (exterior quality), BsmtQual (basement quality)
    + Facilities/additional attributes: e.g., GarageType (garage type), PoolArea (pool area), Fence (fence condition),
      MiscFeature (miscellaneous features)
    + Location/environment attributes: Neighborhood (neighborhood), MSZoning (zoning class), LotShape (lot shape)
    + Categorical & ordinal features: e.g., “Ex”/“Gd”/“TA”/“Fa”/“Po” for quality ratings, which have a natural ordering.

**PRIVACY NOTICE**
---

This application is designed to process the data you provide to generate customized suggestions and results. Your
privacy is paramount.

**We do not collect, store, or transmit your personal information or data.** All processing occurs locally on your
device (in your browser or runtime environment), and **no data is ever sent to an external server or third party.**

- **Local Processing:** Your data never leaves your device. The entire analysis and generation process happens locally.
- **No Data Retention:** Since no data is transmitted, none is stored on any server. Closing the application typically
  clears any temporary local data.
- **Transparency:** The entire codebase is open source. You are encouraged to review the [code](./) to verify how your
  data is handled.

In summary, you maintain full control and ownership of your data at all times.

**LICENCE**
---
This project is open source and available under the **[BSD-3-Clause Licence](LICENCE)**.

In simple terms, this is a very permissive licence that allows you to freely use this code for almost any purpose,
including in proprietary projects, as long as you include the original copyright and licence notice.

Feel free to fork, modify, and build upon this work! We simply ask that you give credit where credit is due.

**ENVIRONMENT SETUP**
---
This project uses **Python 3.12** and [uv](https://docs.astral.sh/uv/) for fast dependency management and virtual
environment handling. The required Python version is automatically detected from the [.python-version](.python-version)
file.

1. **Installing uv**:  
   If you don't have `uv` installed, you can install it using the following command:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # This installation method works on macOS and Linux.
    ```
   Alternatively, you can install it by running the following PowerShell command:
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # This installation method works on Windows.
    ```

   **💡 Recommended**: For the best experience, install `uv` as a standalone tool. Avoid installing it within `pip` or
   `conda` environments to prevent potential dependency conflicts.

2. **Adding Dependencies**:

- To add a main (production) dependency:
    ```bash
    uv add <package_name>
    # This automatically updates pyproject.toml and installs the package
    ```
- To add a development dependency:
    ```bash
    uv add <package_name> --group dev
    # Example: uv add ruff --group dev
    # This adds the package to the [project.optional-dependencies.dev] section automatically
    ```
- To add other types of optional dependencies (e.g., test, docs):
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- To import dependencies from a `requirements.txt` file:
    ```bash
    uv add -r requirements.txt
    # This reads packages from requirements.txt and adds them to pyproject.toml
    ```

3. Removing Dependencies

- To remove a main (production) dependency:
    ```bash
    uv remove <package_name>
    # This automatically updates pyproject.toml and removes the package
    ```
- To remove a development dependency:
    ```bash
    uv remove <package_name> --group dev
    # Example: uv remove ruff --group dev
    # This removes the package from the [project.optional-dependencies.dev] section
    ```
- To remove other types of optional dependencies:
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **Managing the Environment**

- After using add/remove commands, sync the environment:
    ```bash
    uv sync
    ```

**CHANGELOG**
---
This project uses [git-changelog](https://github.com/pawamoy/git-changelog) to automatically generate and maintain a
changelog based on [Conventional Commits](https://www.conventionalcommits.org/).

1. **Installation**
   ```bash
   pip install git-changelog
   # or use uv to add it as a development dependency
   uv add git-changelog --group dev
   ```
2. **Verify Installation**
   ```bash
   pip show git-changelog
   # or check the version specifically
   pip show git-changelog | grep Version
   ```
3. **Configuration**
   Ensure you have a properly configured `pyproject.toml` file at the project root. The configuration should specify
   Conventional Commits as the changelog style. Here is an example configuration:
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **Generate Changelog**
   ```bash
   git-changelog --output CHANGELOG.md
   # Or use uv to run it if installed as a dev dependency
   uv run git-changelog --output CHANGELOG.md
   ```
   This command creates or updates the `CHANGELOG.md` file with all changes based on your git history.
5. **Push Changes**
   ```bash
   git push origin main
   ```
   Alternatively, use your IDE's Git interface (e.g., `Git → Push` in many editors).
6. **Note**:

- The changelog is automatically generated from your commit messages following the Conventional Commits specification.
- Run the generation command whenever you want to update the changelog, typically before a release or after significant
  changes.
