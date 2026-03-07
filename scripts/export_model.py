"""Экспорт модели в ONNX для Triton."""
import os
import sys

# Добавляем директорию scripts в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_onnx import export_explicit_model

# Пути относительно корня проекта
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models/frida_model/1")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")


def main():
    # Создаем директорию, если её нет
    os.makedirs(MODEL_DIR, exist_ok=True)

    export_explicit_model(ONNX_PATH)
    print(f"ONNX сохранён в FP16: {ONNX_PATH}.")


if __name__ == "__main__":
    main()
