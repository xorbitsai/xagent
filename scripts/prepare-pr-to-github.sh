#!/usr/bin/env bash
# 准备 PR 到 https://github.com/xorbitsai/xagent
# 用法: ./scripts/prepare-pr-to-github.sh [你的GitHub用户名]
set -e

GITHUB_USER="${1:-}"
if [[ -z "$GITHUB_USER" ]]; then
  echo "用法: $0 <你的GitHub用户名>"
  echo "示例: $0 laeinxmattrix"
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> 检查 upstream 远程..."
if ! git remote get-url upstream &>/dev/null; then
  git remote add upstream https://github.com/xorbitsai/xagent.git
fi
git fetch upstream

echo "==> 创建 PR 分支 (基于 upstream/main)..."
BRANCH="pr-to-xorbitsai-$(date +%Y%m%d)"
git checkout -B "$BRANCH" upstream/main 2>/dev/null || git checkout "$BRANCH"

echo "==> 合并当前 main 的修改（保留工作区更改）..."
# 将当前 main 的改动 merge 进来（可能有冲突需手动解决）
git merge main -m "Merge local changes for PR" || true

echo ""
echo "==> 请手动完成以下步骤:"
echo "1. 检查 git status，解决可能的冲突"
echo "2. 确认以下敏感信息已隐去:"
echo "   - example.env 中 GOOGLE_API_KEY、GOOGLE_CSE_ID 应为空字符串"
echo "   - 无 .env 文件被添加"
echo "   - 无真实 API Key、密码、Token"
echo "3. 添加 github fork 远程并推送:"
echo "   git remote add github-fork https://github.com/${GITHUB_USER}/xagent.git"
echo "   git push github-fork $BRANCH"
echo "4. 在 GitHub 打开: https://github.com/xorbitsai/xagent/compare/main...${GITHUB_USER}:${BRANCH}"
echo ""
