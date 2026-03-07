#!/usr/bin/env bash
# Экспорт модели FRIDA (если нет) → docker compose (Triton + Encoder). Запуск: ./run.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ONNX_MODEL="models/frida_model/1/model.onnx"
VENV_DIR="triton_env"

# 1. Экспорт модели в ONNX (если ещё нет)
if [ ! -f "$ONNX_MODEL" ]; then
  echo "=== ONNX модели нет, выполняем экспорт в FP16 ==="
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install -q -r requirements.txt
  # Устанавливаем PYTHONPATH чтобы скрипты видели друг друга
  export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts
  python scripts/export_model.py
  deactivate
  echo "=== Удаление виртуального окружения ==="
  rm -rf "$VENV_DIR"
else
  echo "=== ONNX модель уже есть: $ONNX_MODEL ==="
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
echo "  - Метрики: curl http://localhost:8002/metrics"
echo "  - Проверка: curl localhost:8000/v2/models/frida_model"
echo ""
echo "Encoder Service (FastAPI):"
echo "  - API: http://localhost:8080"
echo "  - Health: curl localhost:8080/health"
echo "  - Encode: curl -X POST http://localhost:8080/encode -H 'Content-Type: application/json' -d '{\"query\": \"текст\"}'"
echo ""
echo "Проверка статуса: docker compose ps"
