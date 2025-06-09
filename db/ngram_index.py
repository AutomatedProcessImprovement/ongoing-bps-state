from sqlalchemy import Column, String, Text, Integer, Index, text
from db.database import Base

class NGramIndex(Base):
    __tablename__ = "ngram_index"

    id = Column(Integer, primary_key=True, autoincrement=True)
    process_id = Column(String(36), nullable=False)
    prefix = Column(Text, nullable=False)
    marking = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_process_prefix", "process_id", text("prefix(512)")),
    )
