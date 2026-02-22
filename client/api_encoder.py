"""
Асинхронный клиент для работы с Encoder Service.
Использование аналогично sentence-transformers, но полностью развязан с конкретной реализацией.
"""
import aiohttp
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm


class APIEncoder:
    """
    Клиент для работы с Encoder Service.
    
    Пример использования:
        encoder = APIEncoder(base_url="http://localhost:8080")
        vector = await encoder.encode("Текст для кодирования")
        await encoder.close()
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
    ) -> None:
        """
        Инициализация клиента.
        
        Args:
            base_url: Базовый URL Encoder Service (например, "http://localhost:8080")
            timeout: Таймаут запросов в секундах
        """
        self._session = aiohttp.ClientSession(
            base_url=base_url.rstrip("/") + "/",
            timeout=aiohttp.ClientTimeout(total=timeout),
        )
    
    async def encode(
        self,
        query: Union[str, List[str]],
        prefix: Optional[str] = None,
        batch_size: int = 5,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Кодирование текста(ов) в векторы.
        
        Args:
            query: Текст или список текстов для кодирования
            prefix: Опциональный префикс для текстов
            batch_size: Размер батча для обработки
            show_progress_bar: Показывать ли прогресс-бар
            
        Returns:
            numpy.ndarray: Массив векторов. Для одного текста - 1D массив, для списка - 2D массив
        """
        is_single_query = isinstance(query, str)
        if is_single_query:
            query = [query]
        
        batches = [
            query[i:i + batch_size] 
            for i in range(0, len(query), batch_size)
        ]

        embeddings = []
        for batch in tqdm(batches, desc="Process embeddings", disable=not show_progress_bar):
            async with self._session.post(
                url="encode",
                json={
                    "query": batch,
                    "prefix": prefix,
                    "batch_size": batch_size,
                }
            ) as response:
                response.raise_for_status()
                raw_embeddings = await response.json()
            embeddings.extend(raw_embeddings)

        embeddings = np.array(embeddings)
        if is_single_query:
            embeddings = embeddings[0]
        return embeddings
    
    async def get_sentence_embedding_dimension(self) -> int:
        """
        Получить размерность вектора эмбеддинга.
        
        Returns:
            int: Размерность вектора (1024)
        """
        async with self._session.get(
                url="encode/get_embedding_dimension",
            ) as response:
                response.raise_for_status()
                dimensions = await response.json()
        return dimensions
    
    async def close(self) -> None:
        """Закрыть сессию клиента."""
        await self._session.close()
    
    async def __aenter__(self):
        """Поддержка async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Поддержка async context manager."""
        await self.close()
