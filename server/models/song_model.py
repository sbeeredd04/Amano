from sqlalchemy import Column, Integer, String, BLOB, ForeignKey, Float
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Song(Base):
    __tablename__ = 'songs'
    
    song_id = Column(Integer, primary_key=True)
    track_id = Column(String(22))
    artists = Column(String(100))
    track_name = Column(String(100))
    album_name = Column(String(100))
    track_genre = Column(String(100))
    popularity = Column(Integer)
    features = Column(BLOB)  # Store features as a serialized binary object

class UserHistory(Base):
    __tablename__ = 'user_history'

    user_id = Column(Integer, primary_key=True)
    song_id = Column(Integer, primary_key=True)  # Song ID
    mood = Column(String(20))  # Stores the mood as a string instead of a binary object
    reward = Column(Integer)  # Stores the reward: 1 for liked, -1 for disliked, 0 for neutral

    def __repr__(self):
        return f"<UserHistory(user_id={self.user_id}, song_id={self.song_id}, reward={self.reward}, mood={self.mood})>"

    def serialize(self):
        return {
            'user_id': self.user_id,
            'song_id': self.song_id,
            'reward': self.reward,
            'mood': self.mood  # No need to use pickle anymore for string moods
        }

class UserMood(Base):
    __tablename__ = 'user_mood'

    user_id = Column(Integer, primary_key=True)
    mood = Column(String(20))  # Stores the mood as a string (e.g., 'Happy', 'Sad', etc.)

    def __repr__(self):
        return f"<UserMood(user_id={self.user_id}, mood={self.mood})>"

    def serialize(self):
        return {
            'user_id': self.user_id,
            'mood': self.mood
        }
