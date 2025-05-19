from sqlalchemy import Column, String
from db.database import Base

class NGramIndex(Base):
    __tablename__ = "ngram_index"

    process_id = Column(String(36), primary_key=True)
    prefix = Column(String(255), primary_key=True)
    marking = Column(String(1024), nullable=False)
