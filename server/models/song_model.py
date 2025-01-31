from sqlalchemy import Column, Integer, String, BLOB, ForeignKey, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

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

    song_id = Column(Integer, primary_key=True)  # This should match the song_id from dataset
    track_id = Column(String)  # Keep track_id as a regular column
    artists = Column(String)
    track_name = Column(String)
    album_name = Column(String)
    track_genre = Column(String)
    popularity = Column(Integer)
    features = Column(BLOB)  # Store features as a serialized binary object

    def __repr__(self):
        return f"<Song(song_id={self.song_id}, track_name='{self.track_name}', artists='{self.artists}')>"

    def serialize(self):
        return {
            'song_id': self.song_id,
            'track_id': self.track_id,
            'track_name': self.track_name,
            'artist_name': self.artists,
            'album_name': self.album_name,
            'track_genre': self.track_genre,
            'popularity': self.popularity
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
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    mood = Column(String, nullable=False)

    def __repr__(self):
        return f"<UserMood(user_id='{self.user_id}', mood='{self.mood}')>"

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