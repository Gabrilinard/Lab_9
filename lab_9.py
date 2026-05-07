import time
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os
MINHA_CHAVE = os.getenv("OPENROUTER_API_KEY")

cliente_llm = OpenAI(
    api_key=MINHA_CHAVE,
    base_url="https://openrouter.ai/api/v1",
)