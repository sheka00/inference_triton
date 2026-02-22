"""
Нагрузочный тест обёртки (encoder): POST /encode с текстом.
Запуск: locust -f locustfile.py --host http://localhost:8080
"""
import random
from locust import HttpUser, task, between

SAMPLE_TEXTS = [
    "Договор поставки товара между юридическими лицами.",
    "Исковое заявление о взыскании задолженности.",
    "Трудовой договор с работником на неопределённый срок.",
]


class EncoderUser(HttpUser):
    wait_time = between(0, 0)

    @task(1)
    def encode_one(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/encode",
            json={"query": text, "batch_size": 1},
            name="/encode",
        )
