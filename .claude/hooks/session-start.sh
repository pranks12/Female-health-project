#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR/fern-health"

echo "==> Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo "==> Starting Fern Health API on port 8000..."
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 \
  > /tmp/uvicorn.log 2>&1 &

# Wait for server to be ready
for i in {1..15}; do
  if curl -s http://localhost:8000/api/health &>/dev/null; then
    echo "Server is up and healthy!"
    break
  fi
  sleep 1
done

echo ""
echo "============================================"
echo "  Fern Health API running on port 8000"
echo "  Use the Preview button in Claude Code"
echo "  to open the public URL in your browser."
echo "============================================"
echo ""

# Persist the port for the session
echo "export APP_PORT=8000" >> "$CLAUDE_ENV_FILE"
