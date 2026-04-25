import uuid
import os
import httpx
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Product
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from core.config import settings

router = APIRouter()
qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "products"

class ProductResponse(BaseModel):
    id: str
    name: str
    filepath: str
    filename: str

    class Config:
        from_attributes = True

@router.get("/")
def get_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = db.query(Product).offset(skip).limit(limit).all()
    total = db.query(Product).count()
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@router.post("/")
async def add_item(
    name: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    item_id = str(uuid.uuid4())
    
    # Extract extension or default to .jpg
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".jpg"
        
    filename = f"{item_id}{ext}"
    filepath = f"./sample_images/{filename}"
    
    # Ensure directory exists
    os.makedirs("./sample_images", exist_ok=True)
    
    # Save file
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
        
    # Get embedding from Inference microservice
    files = {"file": (filename, contents, file.content_type)}
    async with httpx.AsyncClient() as http_client:
        try:
            res = await http_client.post(f"{settings.INFERENCE_URL}/embed/image", files=files)
            res.raise_for_status()
            embedding = res.json()["embedding"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
            
    # Save to Qdrant vector database
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=item_id,
                vector=embedding,
                payload={
                    "filename": filename,
                    "filepath": filepath,
                    "name": name
                }
            )
        ]
    )
    
    # Save to Postgres metadata database
    prod = Product(
        id=item_id,
        name=name,
        filepath=filepath,
        filename=filename
    )
    db.add(prod)
    db.commit()
    
    return {"message": "Item added successfully", "id": item_id}
