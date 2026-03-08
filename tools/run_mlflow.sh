#!/bin/bash
set -e

SESSION_NAME="mlflow_ui"

# --- Resolve project directory (parent of tools/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

# --- Load .env directly into environment ---
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "Loaded environment variables from $ENV_FILE"
else
  echo "Warning: .env not found. Environment variables must already be defined."
fi

# --- UV project environment ---
UV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# --- Tracking paths ---
TRACKING_DIR="$PROJECT_DIR/artifacts"
TRACKING_URI="file:${TRACKING_DIR}"
UI_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

# --- Check if required variables exist ---
if [ -z "$MLFLOW_HOST" ] || [ -z "$MLFLOW_PORT" ]; then
  echo "Error: MLFLOW_HOST or MLFLOW_PORT not set in environment."
  exit 1
fi

# --- Ensure the local tracking directory exists ---
mkdir -p "$TRACKING_DIR"

# --- Check if port is free ---
if lsof -i :"$MLFLOW_PORT" &>/dev/null; then
  echo "Error: port $MLFLOW_PORT is already in use."
  exit 1
fi

# --- Launch MLflow in tmux ---
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "MLflow UI is already running in tmux session '$SESSION_NAME'."
else
  echo "Starting MLflow UI on $UI_URI"
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '
    cd \"$PROJECT_DIR\" &&
    uv run --python \"$UV_PYTHON\" mlflow ui \
      --backend-store-uri \"$TRACKING_URI\" \
      --default-artifact-root \"$TRACKING_URI\" \
      --host \"$MLFLOW_HOST\" \
      --port \"$MLFLOW_PORT\"
  '"
  echo "Access MLflow at: $UI_URI"
  echo "Attach to session: tmux attach -t $SESSION_NAME"
fi
