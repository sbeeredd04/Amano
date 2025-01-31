from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from models.song_model import Base, Song
import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
        logging.info("\n=== Initializing Database ===")

        # Add table schema logging
        inspector = inspect(engine)
        columns = inspector.get_columns('songs')
        logging.info("\nTable Schema:")
        for column in columns:
            logging.info(f"- {column['name']}: {column['type']} (Primary Key: {column.get('primary_key', False)})")

        session = get_session()
        try:
            song_count = session.query(Song).count()
            
            if song_count == 0:
                logging.info("\nSongs table is empty. Starting data import...")
                
                # Read first few rows of dataset to verify structure
                df_sample = pd.read_csv(PREPROCESSED_DATA_PATH, nrows=5)
                logging.info("\nDataset sample:")
                logging.info(f"Columns: {df_sample.columns.tolist()}")
                logging.info("\nFirst 5 rows:")
                for _, row in df_sample.iterrows():
                    logging.info(f"\nSong ID: {row['song_id']}")
                    logging.info(f"Track ID: {row['track_id']}")
                    logging.info(f"Track: {row['track_name']}")
                    logging.info(f"Artist: {row['artist_name']}")

                # Import data in chunks
                import_songs_from_csv(session)
                
            else:
                logging.info(f"\nSongs table already populated with {song_count} songs")
                
                # Log sample of existing data
                sample_songs = session.query(Song).limit(5).all()
                logging.info("\nSample of existing songs:")
                for song in sample_songs:
                    logging.info(f"\nSong ID: {song.song_id}")
                    logging.info(f"- Track: {song.track_name}")
                    logging.info(f"- Artist: {song.artists}")
                    logging.info(f"- Genre: {song.track_genre}")
                    
        finally:
            session.close()

    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise

def import_songs_from_csv(session):
    """Import songs from CSV file in chunks."""
    try:
        chunk_size = 1000
        total_rows = sum(1 for _ in open(PREPROCESSED_DATA_PATH)) - 1
        logging.info(f"Total rows in dataset: {total_rows}")

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
            logging.info(f"\nProcessing chunk {chunk_number + 1}...")
            
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
                    
                    if chunk_number == 0 and index < 5:
                        logging.info(f"\nSample song {index + 1}:")
                        logging.info(f"- Song ID: {song.song_id}")
                        logging.info(f"- Track: {song.track_name}")
                        logging.info(f"- Artist: {song.artists}")
                        logging.info(f"- Genre: {song.track_genre}")

                except Exception as e:
                    logging.error(f"Error processing row {index}: {e}")
                    logging.error(f"Row data: {row.to_dict()}")
                    continue

            if songs_to_add:
                session.bulk_save_objects(songs_to_add)
                session.commit()
                total_songs_added += len(songs_to_add)
                logging.info(f"Added {len(songs_to_add)} songs (Total: {total_songs_added})")

            songs_to_add.clear()

        logging.info(f"\nTotal songs added to database: {total_songs_added}")

    except Exception as e:
        logging.error(f"Error importing songs: {str(e)}", exc_info=True)
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
        logging.info(f"\n=== Loading Dataset ===")
        logging.info(f"Total records: {len(df)}")
        logging.info(f"Unique song IDs: {df['song_id'].nunique()}")
        
        # Verify some records match between DB and dataset
        session = get_session()
        try:
            sample_songs = session.query(Song).limit(5).all()
            logging.info("\nVerifying database matches dataset:")
            for song in sample_songs:
                dataset_song = df[df['song_id'] == song.song_id]
                if not dataset_song.empty:
                    logging.info(f"\nSong ID: {song.song_id}")
                    logging.info(f"Database: {song.track_name} by {song.artists}")
                    logging.info(f"Dataset: {dataset_song.iloc[0]['track_name']} by {dataset_song.iloc[0]['artist_name']}")
        finally:
            session.close()
            
        return df
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise

# Initialize database when module is imported
init_db()
