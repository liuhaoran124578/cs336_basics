# Git 与 GitHub 核心工作流指南

本指南旨在提供一个清晰、完整的 Git 和 GitHub 日常使用流程。无论您是初次上传项目，还是进行日常的代码修改与同步，都可以在这里找到对应的操作步骤。

## 核心概念：您的“两个世界”

在开始之前，请牢记这个模型：

1.  **本地仓库 (Local Repository)**：您自己电脑上的项目文件夹。这是您的**私人工作室**，您在这里编写和修改代码。
2.  **远程仓库 (Remote Repository)**：您在 GitHub 上创建的仓库。这是您的**云端展厅和官方档案室**，用于备份、分享和协作。

我们所有的操作，都是为了让这两个“世界”保持同步。

---

## 场景一：首次上传 —— 将全新本地项目发布到 GitHub

这个流程一个项目**只会执行一次**。

### 1. 准备本地项目

在您的本地项目文件夹的终端中，依次执行：

```bash
# 1. 初始化 Git，让 Git 开始管理这个文件夹
git init

# 2. 将所有文件（除了 .gitignore 中指定的）打包到“暂存区”
git add .

# 3. 创建第一个版本快照，并附上说明
git commit -m "Initial commit"
```
> **注意**：如果这是您在这台机器上首次使用 Git，它可能会提示您设置用户名和邮箱。请根据提示运行 `git config --global user.name "Your Name"` 和 `git config --global user.email "you@example.com"`。

### 2. 连接到 GitHub 并推送

在 GitHub 上创建一个**空的**新仓库后，复制页面上 `…or push an existing repository from the command line` 下的命令，并在您的终端中运行：

```bash
# 1. 告诉本地 Git 远程仓库的地址在哪里，并给它取个别名 `origin`
git remote add origin git@github.com:你的用户名/你的仓库名.git

# 2. 将当前分支重命名为 `main` (行业标准)
git branch -M main

# 3. 将本地的 `main` 分支完整地推送到远程 `origin`
git push -u origin main
```

---

## 场景二：日常开发 —— 更新与同步

这是您每天都会用到的核心循环。

### A. 我在本地修改了代码，如何更新到 GitHub？

这是最常见的流程，一个简单的“三步舞曲”。

#### 第1步：检查状态 (`git status`)
养成好习惯，在操作前先查看当前工作区的状态。

```bash
git status
```

#### 第2步：打包修改 (`git add`)
将您想要保存的修改放入“暂存区”。

```bash
# 添加所有被修改或新建的文件
git add .
```

#### 第3步：保存快照 (`git commit`)
为这次修改创建一个本地的版本记录，并写清楚您做了什么。

```bash
# 写一条清晰的 commit 信息，例如：
git commit -m "feat: 在模型中增加了新的激活函数"
```

#### 第44步：推送到云端 (`git push`)
将您在本地创建的所有新提交（commits）一次性上传到 GitHub。

```bash
git push
```

### B. 我在云端修改了文件，如何同步到本地？

比如您直接在 GitHub 网站上编辑了 `README.md`。在本地开始新工作前，必须先将这些云端更新拉取回来。

```bash
git pull
```

这个命令会自动从 GitHub 下载最新的版本，并与您的本地文件合并。

---

## 场景三：日常开发、关机、再开机的标准工作流程

这是一个完整的、从开始到结束的日常工作循环。

#### 1. (开机后) 准备环境
每次重新打开机器，准备开始工作时：

```bash
# 1. 进入您的项目文件夹
cd /path/to/your/project

# 2. 激活 Python 虚拟环境 (如果您的项目使用的话)
source .venv/bin/activate
```

#### 2. (开始编码前) 同步代码
这是一个黄金法则：**永远在最新版本的代码上工作**。

```bash
git pull
```

#### 3. (工作中) 编码与修改
这是您的核心工作时间。您可以自由地修改文件、运行测试、进行实验。

#### 4. (完成小任务后) 保存本地进度
每完成一个独立的功能或修复一个 Bug，就创建一个 `commit`。

```bash
# 1. 检查修改了什么
git status

# 2. 打包所有修改
git add .

# 3. 创建一个清晰的版本快照
git commit -m "fix: 修复了数据集加载时的一个索引错误"
```
> 您可以在一天内进行多次 `commit`，这会让您的版本历史非常干净、有条理。

#### 5. (结束工作或需要分享时) 推送至 GitHub
当您准备结束一天的工作，或者完成了一个需要和他人分享的重要功能时，将所有本地 `commit` 推送到云端。

```bash
git push
```

---

## 常用命令备忘录

| 命令 | 作用 |
| :--- | :--- |
| `git status` | 查看当前工作区的状态，是您最好的朋友。 |
| `git add .` | 将所有修改和新文件添加到暂存区。 |
| `git commit -m "..."` | 创建一个本地的版本快照。 |
| `git push` | 将本地的新提交上传到 GitHub。 |
| `git pull` | 从 GitHub 下载最新版本到本地。 |
| `git clone <url>` | 从远程地址完整地下载一个项目。 |
| `git restore <file>` | 撤销对某个文件的本地修改（在 `commit` 之前）。 |
| `git log` | 查看项目的提交历史记录。 |

```
