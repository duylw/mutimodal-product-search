from fastapi import FastAPI
from api.routes import search

app = FastAPI(title="API Gateway")

app.include_router(search.router, prefix="/search", tags=["search"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
