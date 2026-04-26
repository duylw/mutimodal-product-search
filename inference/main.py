from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load model locally to the container
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Inference Service")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {model_id}")
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down Inference Service")

app = FastAPI(title="Inference Service", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/embed/text")
async def embed_text(req: TextRequest):
    inputs = processor(text=[req.text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'text_embeds'):
            text_features = text_features.text_embeds
        elif not isinstance(text_features, torch.Tensor):
            text_features = text_features[0]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy()[0].tolist()
    return {"embedding": embedding}

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        if hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'image_embeds'):
            image_features = image_features.image_embeds
        elif not isinstance(image_features, torch.Tensor):
            image_features = image_features[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy()[0].tolist()
    return {"embedding": embedding}
