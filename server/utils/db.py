from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from models.song_model import Base
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from models.song_model import Base, Song
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Database setup
DATABASE_URL = 'sqlite:///songs.db'  # Update this to your actual database URL
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Scoped session for thread safety
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Preprocessed dataset path
PREPROCESSED_DATA_PATH = "/Users/sriujjwalreddyb/Amano/spotify_Song_Dataset/final_dataset.csv"
FEATURES = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']

def init_db():
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logging.info("Database initialized successfully.")

        session = get_session()
        song_count = session.query(Song).count()
        
        if song_count == 0:
            logging.info("Songs table is empty. Starting data import...")

            # Load dataset in chunks to manage memory
            chunk_size = 1000
            total_rows = sum(1 for _ in open(PREPROCESSED_DATA_PATH)) - 1
            logging.debug(f"Total rows in dataset: {total_rows}")

            # Columns to extract
            essential_columns = [
                "track_id",
                "artist_name",
                "track_name",
                "album_name",
                "track_genre",
                "popularity"
            ]

            for chunk_number, chunk in enumerate(pd.read_csv(PREPROCESSED_DATA_PATH, usecols=essential_columns, chunksize=chunk_size)):
                logging.info(f"Processing chunk {chunk_number + 1}...")
                
                songs_to_add = []
                for index, row in chunk.iterrows():
                    try:
                        # Log each row for debugging
                        logging.debug(f"Processing row {index}: {row.to_dict()}")
                        
                        # Create Song objects for insertion
                        song = Song(
                            track_id=row["track_id"],
                            artists=str(row["artist_name"]),
                            track_name=str(row["track_name"]),
                            album_name=str(row["album_name"]),
                            track_genre=str(row["track_genre"]),
                            popularity=int(row["popularity"]),
                        )
                        songs_to_add.append(song)
                    except Exception as e:
                        logging.error(f"Error processing row {index}: {e}")
                        continue

                try:
                    session.bulk_save_objects(songs_to_add)
                    session.commit()
                    logging.info(f"Chunk {chunk_number + 1} committed with {len(songs_to_add)} songs.")
                except Exception as e:
                    logging.error(f"Error committing chunk: {e}")
                    session.rollback()
                
                # Clear memory
                songs_to_add.clear()

            # Log total rows in the songs table after insertion
            final_song_count = session.query(Song).count()
            logging.info(f"Data import completed. Total songs in the database: {final_song_count}")
        else:
            logging.info(f"Songs table already populated with {song_count} songs.")

    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        session.close()


# Utility function to get a database session
def get_session():
    """
    Provide a scoped session for database interactions.
    Make sure to close the session after use.
    """
    try:
        session = Session()
        logging.debug("Database session created.")
        return session
    except Exception as e:
        logging.error(f"Error creating database session: {e}")
        raise

# Close session
def close_session(session):
    """
    Close a given session to avoid resource leaks.
    """
    try:
        session.close()
        logging.debug("Database session closed.")
    except Exception as e:
        logging.error(f"Error closing session: {e}")
