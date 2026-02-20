#!/usr/bin/env bash
# Конвертация ONNX в TensorRT через trtexec в Docker.
# Использование: ./convert_trt.sh [путь к ONNX] [путь к model.plan]
# По умолчанию: model.onnx -> models/bge_model/1/model.plan

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${1:-$SCRIPT_DIR/model.onnx}"
TRT_OUTPUT="${2:-$SCRIPT_DIR/models/bge_model/1/model.plan}"

mkdir -p "$(dirname "$TRT_OUTPUT")"

docker run --gpus all --rm \
  -v "$SCRIPT_DIR:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tensorrt:23.09-py3 \
  trtexec \
  "--onnx=$ONNX_PATH" \
  "--saveEngine=$TRT_OUTPUT" \
  --workspace=4096
