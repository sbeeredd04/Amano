from sqlalchemy import Column, Integer, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from models.song_model import Base
from datetime import datetime

class RecommendationPool(Base):
    __tablename__ = 'recommendation_pools'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    recommendation_pool = Column(JSON, nullable=False)  # Store the full pool of recommendations
    user_songs_pool = Column(JSON, nullable=False)  # Store user songs pool
    popular_songs_pool = Column(JSON, nullable=True)  # Make sure this exists
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def serialize(self):
        return {
            'user_id': self.user_id,
            'recommendation_pool': self.recommendation_pool,
            'user_songs_pool': self.user_songs_pool,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 