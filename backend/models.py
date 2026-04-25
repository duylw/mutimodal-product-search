from sqlalchemy import Column, String
from database import Base
import uuid

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, index=True)
    filepath = Column(String)
    filename = Column(String)
