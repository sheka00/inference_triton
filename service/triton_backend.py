"""Асинхронный aiohttp-клиент к Triton с поддержкой батчинга."""
from __future__ import annotations

import aiohttp
from typing import List

TRITON_MODEL = "bge_model"
MAX_LENGTH = 512


class TritonInferClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def infer_batch(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
    ) -> List[List[float]]:
        """
        Отправляет батч в Triton одним запросом.
        input_ids: [batch_size, MAX_LENGTH]
        attention_mask: [batch_size, MAX_LENGTH]
        """
        batch_size = len(input_ids)
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [batch_size, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": input_ids,
                },
                {
                    "name": "attention_mask",
                    "shape": [batch_size, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": attention_mask,
                },
            ]
        }
        url = f"{self._base_url}/v2/models/{TRITON_MODEL}/infer"
        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            
        out = data.get("outputs", [])
        if not out or out[0].get("name") != "output":
            raise ValueError("Unexpected Triton response shape")
            
        # Triton возвращает плоский список или вложенный в зависимости от версии/настроек.
        # Для bge_model выход [batch_size, 1024].
        raw_data = out[0]["data"]
        # Группируем обратно в [batch_size, 1024]
        embedding_dim = 1024
        embeddings = [raw_data[i : i + embedding_dim] for i in range(0, len(raw_data), embedding_dim)]
        return embeddings

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()