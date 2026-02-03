# 推送 Weak-Driving Learning 到 GitHub 私密仓库

## 准备工作

本地 Git 仓库已初始化并完成首次提交，包含所有项目文件。

## 方法一：使用脚本（推荐）

直接运行：
```bash
bash push_to_github.sh
```

脚本会引导您完成所有步骤。

## 方法二：手动操作

### 步骤 1: 在 GitHub 上创建私密仓库

1. 访问 https://github.com/new
2. 填写信息：
   - **Repository name**: `Weak-Driving-Learning`（或您喜欢的名称）
   - **Visibility**: 选择 **Private**（私密）
   - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
   - **不要**添加 .gitignore 或 LICENSE（我们已经有了）
3. 点击 **Create repository**

### 步骤 2: 添加远程仓库并推送

在项目目录下运行（替换 `YOUR_USERNAME` 为您的 GitHub 用户名，`REPO_NAME` 为仓库名）：

```bash
cd "/root/buaa/czh/Weak-Driving Learning"

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 推送代码
git push -u origin main
```

如果使用 SSH（需要配置 SSH key）：
```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

### 步骤 3: 验证

访问 `https://github.com/YOUR_USERNAME/REPO_NAME` 确认代码已成功推送。

## 方法三：使用 GitHub CLI（如果已安装）

如果您安装了 GitHub CLI (`gh`)，可以运行：

```bash
cd "/root/buaa/czh/Weak-Driving Learning"
gh repo create Weak-Driving-Learning --private --source=. --remote=origin --push
```

## 注意事项

- 确保您已登录 GitHub 账户
- 如果使用 HTTPS，推送时可能需要输入 GitHub 用户名和 Personal Access Token（不是密码）
- 如果使用 SSH，确保已配置 SSH key
