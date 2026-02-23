import aiohttp
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm


class APIEncoder:
    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
    ) -> None:
        self._session = aiohttp.ClientSession(
            base_url=base_url.rstrip("/") + "/",
            timeout=aiohttp.ClientTimeout(float(timeout)),
        )

    async def health(self) -> dict:
        async with self._session.get(url="/health") as response:
            response.raise_for_status()
            return await response.json()

    async def encode(
        self,
        query: Union[str, List[str]],
        prefix: Optional[str] = None,
        batch_size: int = 5,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
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
                url="/encode",
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
    
    async def get_vector_dim(self) -> int:
        async with self._session.get(url="/get_vector_dim") as response:
            response.raise_for_status()
            return await response.json()
    
    async def close(self) -> None:
        await self._session.close()
