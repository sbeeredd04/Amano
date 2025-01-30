from sqlalchemy import Column, Integer, String, BLOB, ForeignKey  # Added ForeignKey import
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    image = Column(String(255), nullable=True)  # Optional for profile picture

    def __repr__(self):
        return f"<User(user_id={self.user_id}, email={self.email}, name={self.name})>"

    def serialize(self):
        return {
            'user_id': self.user_id,
            'email': self.email,
            'name': self.name,
            'image': self.image,
        }


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

    def serialize(self):
        """Convert the song object to a dictionary for JSON serialization"""
        return {
            'song_id': self.song_id,
            'track_id': self.track_id,
            'artists': self.artists,
            'track_name': self.track_name,
            'album_name': self.album_name,
            'track_genre': self.track_genre,
            'popularity': self.popularity
            # Excluding features as it's a BLOB
        }

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
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    mood = Column(String(20))  # Stores the mood as a string (e.g., 'Happy', 'Sad', etc.)

    def __repr__(self):
        return f"<UserMood(user_id={self.user_id}, mood={self.mood})>"

    def serialize(self):
        return {
            'user_id': self.user_id,
            'mood': self.mood
        }

class Playlist(Base):
    __tablename__ = 'playlists'

    playlist_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)  # ForeignKey to a User table if it exists
    name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<Playlist(playlist_id={self.playlist_id}, user_id={self.user_id}, name={self.name})>"

    def serialize(self):
        return {
            'playlist_id': self.playlist_id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description
        }


class PlaylistSong(Base):
    __tablename__ = 'playlist_songs'

    id = Column(Integer, primary_key=True)
    playlist_id = Column(Integer, ForeignKey('playlists.playlist_id'), nullable=False)
    song_id = Column(Integer, ForeignKey('songs.song_id'), nullable=False)

    def __repr__(self):
        return f"<PlaylistSong(playlist_id={self.playlist_id}, song_id={self.song_id})>"

    def serialize(self):
        return {
            'playlist_id': self.playlist_id,
            'song_id': self.song_id
        }