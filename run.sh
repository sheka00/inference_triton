#!/usr/bin/env bash
# Экспорт модели (если нет) → docker compose (Triton + Encoder). Запуск: ./run.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRT_MODEL="models/bge_model/1/model.plan"
VENV_DIR="triton_env"

# 1. Экспорт модели в TensorRT (если ещё нет)
if [ ! -f "$TRT_MODEL" ]; then
  echo "=== TRT модели нет, выполняем экспорт ==="
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install -q -r requirements.txt
  python scripts/export_model.py
  ./scripts/convert_trt.sh model.onnx "$TRT_MODEL"
  rm -f model.onnx model.onnx.data
  deactivate
  echo "=== Удаление виртуального окружения ==="
  rm -rf "$VENV_DIR"
else
  echo "=== TRT модель уже есть: $TRT_MODEL ==="
fi

# 2. Остановить существующие контейнеры (если есть)
echo "=== Останавливаем существующие контейнеры ==="
docker compose down 2>/dev/null || true

# 3. Запуск Triton Server + Encoder Service через Docker Compose
echo "=== Запуск Triton Server + Encoder Service ==="
docker compose up -d --build

echo ""
echo "=== Сервисы запущены ==="
echo "Triton Server:"
echo "  - HTTP: http://localhost:8000"
echo "  - gRPC: http://localhost:8001"
echo "  - Метрики: http://localhost:8002"
echo "  - Проверка: curl localhost:8000/v2/models/bge_model"
echo ""
echo "Encoder Service (FastAPI):"
echo "  - API: http://localhost:8080"
echo "  - Health: curl localhost:8080/health"
echo "  - Encode: curl -X POST http://localhost:8080/encode -H 'Content-Type: application/json' -d '{\"query\": \"текст\"}'"
echo ""
echo "Проверка статуса: docker compose ps"
