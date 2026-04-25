from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from PIL import Image
import io

# Setup model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    # Qdrant client connection (points to the local file DB we created)
    client = QdrantClient(path="./qdrant_data")
    yield
    if client is not None:
        client.close()

app = FastAPI(title="Visual Search API", lifespan=lifespan)

COLLECTION_NAME = "products"

class TextQuery(BaseModel):
    query: str

@app.post("/search/text")
async def search_by_text(data: TextQuery):
    # Encode text into a vector using CLIP
    inputs = processor(text=[data.query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy()[0].tolist()
        
    # Search the vector database
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=3
    ).points
    
    return {"results": [{"score": hit.score, "item": hit.payload} for hit in search_result]}

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Encode image into a vector using CLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        if hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy()[0].tolist()
        
    # Search the vector database
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=3
    ).points
    
    return {"results": [{"score": hit.score, "item": hit.payload} for hit in search_result]}