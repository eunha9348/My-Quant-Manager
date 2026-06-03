#!/bin/bash
set -euo pipefail

# 웹 환경에서만 실행
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "퀀트 매니저 환경 설치 중..."

pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt" --quiet

# PYTHONPATH 설정 (프로젝트 루트에서 모듈 import 가능하게)
echo 'export PYTHONPATH="$CLAUDE_PROJECT_DIR"' >> "$CLAUDE_ENV_FILE"

echo "환경 설치 완료"
