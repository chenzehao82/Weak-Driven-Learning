#!/bin/bash

# 脚本：将 Weak-Driving Learning 推送到 GitHub 私密仓库
# 使用方法：bash push_to_github.sh

set -e

echo "=========================================="
echo "推送 Weak-Driving Learning 到 GitHub 私密仓库"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "错误：请在项目根目录运行此脚本"
    exit 1
fi

# 获取 GitHub 用户名
read -p "请输入您的 GitHub 用户名: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "错误：GitHub 用户名不能为空"
    exit 1
fi

# 获取仓库名称（默认使用项目文件夹名）
DEFAULT_REPO_NAME="Weak-Driving-Learning"
read -p "请输入仓库名称（直接回车使用默认: $DEFAULT_REPO_NAME）: " REPO_NAME
REPO_NAME=${REPO_NAME:-$DEFAULT_REPO_NAME}

echo ""
echo "配置信息："
echo "  GitHub 用户名: $GITHUB_USERNAME"
echo "  仓库名称: $REPO_NAME"
echo "  仓库类型: 私密仓库"
echo ""

read -p "确认创建并推送？(y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "步骤 1: 检查 Git 状态..."
if [ -n "$(git status --porcelain)" ]; then
    echo "警告：有未提交的更改，正在添加..."
    git add .
    git commit -m "Update before push to GitHub"
fi

echo ""
echo "步骤 2: 检查远程仓库..."
if git remote | grep -q "^origin$"; then
    echo "已存在 origin 远程仓库，正在移除..."
    git remote remove origin
fi

echo ""
echo "步骤 3: 添加 GitHub 远程仓库..."
GITHUB_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
git remote add origin "$GITHUB_URL"

echo ""
echo "=========================================="
echo "下一步操作："
echo "=========================================="
echo ""
echo "1. 请在浏览器中访问：https://github.com/new"
echo "2. 创建新仓库："
echo "   - Repository name: $REPO_NAME"
echo "   - 选择 'Private'（私密）"
echo "   - 不要初始化 README、.gitignore 或 LICENSE（我们已经有了）"
echo "   - 点击 'Create repository'"
echo ""
echo "3. 创建完成后，按回车继续推送代码..."
read -p ""

echo ""
echo "步骤 4: 推送代码到 GitHub..."
echo "正在推送到: $GITHUB_URL"
git push -u origin main

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
echo "仓库地址: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
