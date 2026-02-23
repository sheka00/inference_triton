"""Пример: python examples/example_usage.py (из корня проекта)."""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.api_encoder import APIEncoder


async def main():
    encoder = APIEncoder(base_url="http://localhost:8080")
    try:
        print("Сервис:", await encoder.health())
        print("Размерность вектора:", await encoder.get_vector_dim())
        
        # Закодировать один текст
        print("\n=== Кодирование одного текста ===")
        text = "Договор поставки товара между юридическими лицами."
        vector = await encoder.encode(text)
        print(f"Текст: {text}")
        print(f"Вектор формы: {vector.shape}")
        print(f"Первые 5 значений: {vector[:5]}")
        
        # Закодировать несколько текстов
        print("\n=== Кодирование нескольких текстов ===")
        texts = [
            "Договор поставки товара между юридическими лицами.",
            "Исковое заявление о взыскании задолженности.",
            "Трудовой договор с работником на неопределённый срок.",
        ]
        vectors = await encoder.encode(texts, batch_size=2, show_progress_bar=True)
        print(f"Количество текстов: {len(texts)}")
        print(f"Векторы формы: {vectors.shape}")
        
        # Использование с префиксом
        print("\n=== Кодирование с префиксом ===")
        vector_with_prefix = await encoder.encode(
            "договор",
            prefix="query:",
            batch_size=1
        )
        print(f"Вектор с префиксом формы: {vector_with_prefix.shape}")
        
    finally:
        await encoder.close()


if __name__ == "__main__":
    print("=== Пример использования APIEncoder ===\n")
    asyncio.run(main())
