import asyncio
import numpy as np
import torch
import warnings
from sentence_transformers import SentenceTransformer
from client import APIEncoder

# Подавляем предупреждения о несовместимости CUDA
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

def get_sentence_transformer_with_compatibility(model_name: str):
    """Загружает SentenceTransformer с проверкой совместимости GPU"""
    try:
        model = SentenceTransformer(model_name)
        
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            print(f"Найден GPU: {device.name} (CUDA capability {device.major}.{device.minor})")
            
            if device.major < 7:
                print(f"⚠ GPU не поддерживается, переключаемся на CPU...")
                model = model.to('cpu')
            else:
                model = model.to('cuda')
        else:
            model = model.to('cpu')
    except Exception as e:
        print(f"⚠ Ошибка, используем CPU: {e}")
        model = SentenceTransformer(model_name, device='cpu')
    
    return model

async def main():
    # Инициализация клиентов
    encoder_triton = APIEncoder("http://localhost:8080")
    encoder = get_sentence_transformer_with_compatibility("ai-forever/FRIDA")
    
    texts = [
        "Привет как дела",
        "Это мог бы быть большой большой текст, но это не такой ольшой большой текст",
        "кошечки и пёсики",
        "пёсики и кошечки",
        "гавно залупа пенис хер",
        "во имя отца сына и святого духа",
        "жили были пацаны, громко слушали басы",
        "тест тест тестовый тестянский тест",
    ]
    
    # Получаем эмбеддинги
    emb1 = encoder.encode(texts, prompt="search_query: ")
    emb2 = await encoder_triton.encode(texts, prefix="search_query: ")
    
    # Подсчет скалярных произведений (ровно как в примере)
    print("\nРезультаты:")
    print("np.sum(emb1 * emb1, axis=1):", np.sum(emb1 * emb1, axis=1))
    print("np.sum(emb2 * emb2, axis=1):", np.sum(emb2 * emb2, axis=1))
    print("np.sum(emb1 * emb2, axis=1):", np.sum(emb1 * emb2, axis=1))
    
    await encoder_triton.close()

if __name__ == "__main__":
    asyncio.run(main())
