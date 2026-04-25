from fastapi import FastAPI
from api.routes import search, items
from database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="API Gateway")

app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(items.router, prefix="/items", tags=["items"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
