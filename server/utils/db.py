from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from models.song_model import Base, Song
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Database setup
DATABASE_URL = 'sqlite:///songs.db'
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Scoped session for thread safety
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Preprocessed dataset path
PREPROCESSED_DATA_PATH = "/Users/sriujjwalreddyb/Amano/spotify_Song_Dataset/final_dataset.csv"

# Global DataFrame instance
_df_scaled = None

def get_session():
    return Session()

def init_db():
    """Initialize database and load songs data."""
    try:
        # Create all tables first
        Base.metadata.create_all(engine)

        # Add table schema logging
        inspector = inspect(engine)
        columns = inspector.get_columns('songs')

        session = get_session()
        try:
            song_count = session.query(Song).count()
            
            if song_count == 0:
                # Read first few rows of dataset to verify structure
                df_sample = pd.read_csv(PREPROCESSED_DATA_PATH, nrows=5)

                # Import data in chunks
                import_songs_from_csv(session)
                    
        finally:
            session.close()

    except Exception as e:
        raise

def import_songs_from_csv(session):
    """Import songs from CSV file in chunks."""
    try:
        chunk_size = 1000
        total_rows = sum(1 for _ in open(PREPROCESSED_DATA_PATH)) - 1

        essential_columns = [
            "song_id",
            "track_id",
            "artist_name",
            "track_name",
            "album_name",
            "track_genre",
            "popularity"
        ]

        total_songs_added = 0
        
        for chunk_number, chunk in enumerate(pd.read_csv(PREPROCESSED_DATA_PATH, usecols=essential_columns, chunksize=chunk_size)):
            songs_to_add = []
            for index, row in chunk.iterrows():
                try:
                    song = Song(
                        song_id=int(row["song_id"]),
                        track_id=str(row["track_id"]),
                        artists=str(row["artist_name"]),
                        track_name=str(row["track_name"]),
                        album_name=str(row["album_name"]),
                        track_genre=str(row["track_genre"]),
                        popularity=int(row["popularity"])
                    )
                    songs_to_add.append(song)

                except Exception as e:
                    continue

            if songs_to_add:
                session.bulk_save_objects(songs_to_add)
                session.commit()
                total_songs_added += len(songs_to_add)

            songs_to_add.clear()

    except Exception as e:
        session.rollback()
        raise

def get_dataset():
    """Get the singleton instance of the dataset."""
    global _df_scaled
    if _df_scaled is None:
        _df_scaled = load_dataset()
    return _df_scaled

def load_dataset():
    """Load and preprocess the dataset."""
    try:
        # Initialize database first if needed
        init_db()
        
        # Now load the dataset
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        
        # Verify some records match between DB and dataset
        session = get_session()
        try:
            sample_songs = session.query(Song).limit(5).all()
            for song in sample_songs:
                dataset_song = df[df['song_id'] == song.song_id]
        finally:
            session.close()
            
        return df
        
    except Exception as e:
        raise

# Initialize database when module is imported
init_db()
