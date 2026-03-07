"""Экспорт BGE-M3 в ONNX: веса в FP16, выход в FP32 для совместимости с JSON."""
import torch
import warnings
import os
import onnx
from transformers import AutoModel, AutoTokenizer

from model_wrapper import BGE_M3_Wrapper

warnings.filterwarnings("ignore", category=Warning)


def export_explicit_model(onnx_path="model.onnx"):
    print("Загрузка модели...")
    model = AutoModel.from_pretrained("Roflmax/bge-m3-legal-ru-updata")
    tokenizer = AutoTokenizer.from_pretrained("Roflmax/bge-m3-legal-ru-updata")
    
    # Конвертируем веса в FP16 для скорости на GPU
    print("Конвертация весов в FP16...")
    model = model.half()
    model.eval()

    # Обертка уже содержит нормализацию, которая вернет FP16, 
    # если модель в FP16. Нам нужно явно привести выход к FP32.
    class FP32_Output_Wrapper(torch.nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.wrapper = wrapper
        def forward(self, input_ids, attention_mask):
            out = self.wrapper(input_ids, attention_mask)
            return out.float() # Принудительно в FP32 для Triton JSON

    wrapper = FP32_Output_Wrapper(BGE_M3_Wrapper(model))
    wrapper.eval()

    vocab_size = len(tokenizer)
    dummy_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    dummy_mask = torch.ones((1, 512), dtype=torch.long)
    
    print(f"Экспорт модели в {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    # Исправляем IR version до 9
    print(f"Исправление IR version до 9...")
    model_onnx = onnx.load(onnx_path)
    model_onnx.ir_version = 9
    onnx.save(model_onnx, onnx_path)

    print(f"Модель успешно экспортирована (веса FP16, выход FP32): {onnx_path}")
    return onnx_path
