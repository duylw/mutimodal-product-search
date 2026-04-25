import os
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Directories
DATA_DIR = "./sample_images"

# Use env vars so it works both locally and inside Docker
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")

def wait_for_services():
    print("Waiting for Qdrant and Inference services to be ready...")
    for _ in range(30):
        try:
            # Check if inference API is up
            requests.get(f"{INFERENCE_URL}/docs", timeout=2)
            # Check if Qdrant is up
            QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2).get_collections()
            print("Services are ready!")
            return True
        except Exception:
            print("Waiting for services...")
            time.sleep(3)
    return False

SAMPLE_IMAGES = {
    "red_jacket.jpg": "https://images.unsplash.com/photo-1551028719-00167b16eac5?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
    "blue_jeans.jpg": "https://images.unsplash.com/photo-1542272604-787c3835535d?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
    "white_sneakers.jpg": "https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
    "sunglasses.jpg": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
    "black_backpack.jpg": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60", 
}

def download_images():
    print("Downloading sample images...")
    for filename, url in SAMPLE_IMAGES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as f:
                f.write(requests.get(url).content)
    print("Download complete.")

def index_images():
    if not wait_for_services():
        print("Services did not become ready in time. Exiting seeder.")
        return
        
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    COLLECTION_NAME = "products"

    # Check if already seeded (Idempotency)
    try:
        collections = client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            count = client.count(collection_name=COLLECTION_NAME).count
            if count > 0:
                print(f"Database already seeded with {count} items. Skipping seeding.")
                return
    except Exception as e:
        print(f"Error checking collections: {e}")

    print("Creating collection...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    print(f"Reading images from {DATA_DIR}...")
    points = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found. Cannot seed.")
        return
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    if not files:
        print("No images found to seed!")
        return
        
    for i, filename in enumerate(files):
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, "rb") as f:
            req_files = {"file": (filename, f, "image/jpeg")}
            try:
                res = requests.post(f"{INFERENCE_URL}/embed/image", files=req_files)
                res.raise_for_status()
                embedding = res.json()["embedding"]
            except Exception as e:
                print(f"Failed to embed {filename}: {e}")
                continue
            
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "filename": filename,
                "filepath": f"./sample_images/{filename}",
                "name": filename.replace(".jpg", "").replace("_", " ").title()
            }
        )
        points.append(point)
        print(f"Embedded {filename}")
        
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Seeded {len(points)} images successfully!")

if __name__ == "__main__":
    download_images()
    index_images()
