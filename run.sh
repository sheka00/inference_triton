#!/usr/bin/env bash
# Полный цикл: среда → зависимости → экспорт модели (если нет) → удаление среды → запуск Triton.
# Запуск: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRT_MODEL="models/bge_model/1/model.plan"
VENV_DIR="triton_env"
TRITON_CONTAINER="triton_server"

# 1. Экспорт модели в TensorRT (если ещё нет)
if [ ! -f "$TRT_MODEL" ]; then
  echo "=== TRT модели нет, выполняем экспорт ==="
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install -q -r requirements.txt
  python export_model.py
  ./convert_trt.sh model.onnx "$TRT_MODEL"
  rm -f model.onnx model.onnx.data
  deactivate
  echo "=== Удаление виртуального окружения ==="
  rm -rf "$VENV_DIR"
else
  echo "=== TRT модель уже есть: $TRT_MODEL ==="
fi

# 2. Остановить старый контейнер Triton (если есть)
if docker ps -q -f name="^${TRITON_CONTAINER}$" 2>/dev/null | grep -q .; then
  echo "=== Останавливаем существующий контейнер Triton ==="
  docker stop "$TRITON_CONTAINER" 2>/dev/null || true
  docker rm "$TRITON_CONTAINER" 2>/dev/null || true
fi

# 3. Запуск Triton Server
echo "=== Запуск Triton Server ==="
docker run -d --name "$TRITON_CONTAINER" --gpus=all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$SCRIPT_DIR/models:/models" \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

echo "Готово. Порты: 8000 (HTTP), 8001 (gRPC), 8002 (метрики). Проверка: curl localhost:8000/v2/models/bge_model"
