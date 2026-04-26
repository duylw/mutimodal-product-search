from fastapi import FastAPI
from api.routes import search, items
from database import engine, Base
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup and shutdown."""
    logger.info("Application starting up")
    logger.info("Checking database connection")
    try:
        # Test database connection
        with engine.connect() as conn:
           pass
        logger.info("Database connection successful")
        logger.info("Indexing sample images...")
        
        # Run indexer script
        from indexer import index_sample_images
        await index_sample_images()
        logger.info("Sample images indexed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
    
    yield
    
    logger.info("Application shutting down")

app = FastAPI(title="API Gateway", lifespan=lifespan)

app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(items.router, prefix="/items", tags=["items"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
