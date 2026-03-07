"""Экспорт ai-forever/FRIDA в ONNX с квантизацией."""
import torch
import warnings
import os
import onnx
from transformers import T5EncoderModel, AutoTokenizer

from model_wrapper import FRIDA_Wrapper

warnings.filterwarnings("ignore", category=Warning)


def export_explicit_model(onnx_path="model.onnx"):
    print("Загрузка модели...")
    model_id = "ai-forever/FRIDA"
    model = T5EncoderModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
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

    wrapper = FP32_Output_Wrapper(FRIDA_Wrapper(model))
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
        opset_version=18, # Современный opset для современных Triton
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    # Исправление IR version до 9 для совместимости с Triton
    print(f"Исправление IR version до 9...")
    model_onnx = onnx.load(onnx_path)
    model_onnx.ir_version = 9
    onnx.save(model_onnx, onnx_path)

    print(f"Модель успешно экспортирована в FP16 и пропатчена до IR 9: {onnx_path}")
    return onnx_path
