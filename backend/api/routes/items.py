from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Product
from pydantic import BaseModel

router = APIRouter()

class ProductResponse(BaseModel):
    id: int
    name: str
    filepath: str
    filename: str

    class Config:
        orm_mode = True

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
