# src/models.py
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class LLM:
    def __init__(self, model: str):
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.2):
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
