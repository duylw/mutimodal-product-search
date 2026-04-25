from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import httpx
from qdrant_client import QdrantClient
from core.config import settings

router = APIRouter()
client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "products"

class TextQuery(BaseModel):
    query: str

@router.post("/text")
async def search_text(data: TextQuery):
    # 1. Call inference service to get the embedding for the text
    async with httpx.AsyncClient() as http_client:
        try:
            res = await http_client.post(f"{settings.INFERENCE_URL}/embed/text", json={"text": data.query})
            res.raise_for_status()
            embedding = res.json()["embedding"]
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Inference service unavailable: {e}")
    
    # 2. Query Qdrant vector database
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=3
        ).points
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    return {"results": [{"score": hit.score, "item": hit.payload} for hit in search_result]}

@router.post("/image")
async def search_image(file: UploadFile = File(...)):
    # 1. Call inference service to get embedding for the image
    contents = await file.read()
    files = {"file": (file.filename, contents, file.content_type)}
    
    async with httpx.AsyncClient() as http_client:
        try:
            res = await http_client.post(f"{settings.INFERENCE_URL}/embed/image", files=files)
            res.raise_for_status()
            embedding = res.json()["embedding"]
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Inference service unavailable: {e}")
        
    # 2. Query Qdrant vector database
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=3
        ).points
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    return {"results": [{"score": hit.score, "item": hit.payload} for hit in search_result]}
