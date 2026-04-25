import os

class Settings:
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")

settings = Settings()
