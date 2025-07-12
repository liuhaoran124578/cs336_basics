# 项目环境配置指南 

## 第 1 步：安装基础工具

在开始之前，请确保您的新机器上已经安装了 `git`。然后，我们需要安装 `uv`。

### 1.1 安装 uv

`uv` 是一个由 Rust 编写的、极速的 Python 包管理器。请在您的终端中运行以下命令来安装它：

```bash
pip install uv
```
---

## 第 2 步：配置 GitHub SSH 密钥（一次性设置）

为了让您的新机器能够安全地从 GitHub 克隆（clone）和推送（push）代码，我们需要为其配置一把专属的“钥匙”（SSH 密钥）。**每台新机器都需要执行一次此步骤。**

### 2.1 生成新的 SSH 密钥

在终端中运行以下命令。**请务必将引号中的邮箱地址替换为您自己注册 GitHub 时使用的邮箱。**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

接下来，系统会向您提问。您无需输入任何内容，只需**连续按三次 `Enter` 键**接受所有默认设置即可。

### 2.2 将公钥添加到您的 GitHub 账户

1.  **查看并复制您的公钥**（公钥是钥匙的“公开”部分，可以安全地分享）。

    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    这条命令会显示一长串以 `ssh-ed25519` 开头的文本。请使用鼠标**从头到尾、完整地**复制这一整串文本。

2.  **登录 GitHub 网站，添加公钥**：
    *   进入您的 GitHub **Settings** (点击右上角头像 -> Settings)。
    *   在左侧菜单栏中，点击 **SSH and GPG keys**。
    *   点击绿色的 **New SSH key** 按钮。
    *   **Title**: 给这把新钥匙起一个能区分的名字，比如 `My New AutoDL Machine`。
    *   **Key**: 将您刚才复制的那一整串公钥，完整地粘贴到这个大的输入框里。
    *   点击 **Add SSH key**。

### 2.3 测试连接

回到您的终端，运行以下命令来测试“钥匙”是否能正常开门：

```bash
ssh -T git@github.com
```

如果看到包含您 GitHub 用户名的欢迎信息（如 `Hi liuhaoran124578! You've successfully authenticated...`），则证明配置成功！

---

## 第 3 步：克隆项目并配置环境

现在您的机器已经准备就绪，可以开始配置项目本身了。

### 3.1 克隆您的 GitHub 仓库

使用 `git clone` 命令将项目文件从 GitHub 下载到您的本地机器。

```bash
git clone git@github.com:liuhaoran124578/cs336_basics.git
```

### 3.2 进入项目目录

```bash
cd cs336_basics/
```

### 3.3 创建并激活虚拟环境

我们将创建一个独立的、隔离的 Python 环境来运行此项目。

1.  **创建虚拟环境**
    `uv` 会自动读取 `pyproject.toml` 文件中的 `requires-python = "==3.10"`，并使用 Python 3.10 来创建环境。

    ```bash
    uv venv
    ```
    这会在当前目录下创建一个名为 `.venv` 的文件夹。

2.  **激活虚拟环境**
    ```bash
    source .venv/bin/activate
    ```
    激活后，您的命令提示符左侧会出现 `(.venv)` 字样。

### 3.4 同步并安装所有依赖

这是最关键的一步。我们将使用一条命令来完美复原环境。

```bash
uv pip install -e ".[dev]"
```

这条智能的命令会：
1.  首先检查是否存在 `uv.lock` 文件。
2.  **如果存在，它将严格按照 `uv.lock` 文件中的精确版本来安装所有依赖**，确保环境 100% 可复现。
3.  如果不存在，它才会根据 `pyproject.toml` 来安装。
4.  `-e ".[dev]"` 参数会同时安装项目本身（可编辑模式）和所有开发工具（如 `ruff` 和 `pytest`）。

---
