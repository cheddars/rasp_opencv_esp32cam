from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, CHAR, Float, DateTime, func, Index
from sqlalchemy.dialects.mysql import FLOAT
from database import Base


class ImageArchive(Base):
    __tablename__ = "image_archive"
    id = Column(Integer, primary_key=True, autoincrement=True)
    yyyy = Column(CHAR(4))
    mm = Column(CHAR(2))
    dd = Column(CHAR(2))
    module = Column(String(50))
    image_path = Column(String(255))
    detection = Column(String(100))
    probability = Column(FLOAT(precision=5, scale=2))
    svr_dt = Column(DateTime, server_default=func.now(), index=True)
    __table_args__ = (Index('image_archive_ymd_idx', "yyyy", "mm", "dd"),)
