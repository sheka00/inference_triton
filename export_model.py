"""
Точка входа: экспорт модели в ONNX.
Конвертацию в TensorRT выполняет run.sh через convert_trt.sh.
"""
import os

from export_onnx import export_explicit_model

TRT_MODEL_PATH = "models/bge_model/1/model.plan"
ONNX_PATH = "model.onnx"


def main():
    if os.path.exists(TRT_MODEL_PATH):
        print(f"TRT модель уже есть: {TRT_MODEL_PATH}. Экспорт пропущен.")
        return

    export_explicit_model(ONNX_PATH)
    print(f"ONNX сохранён: {ONNX_PATH}. Дальше run.sh вызовет convert_trt.sh.")


if __name__ == "__main__":
    main()
