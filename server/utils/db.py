# server/utils/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.song_model import Base

# Database setup
engine = create_engine('sqlite:///songs.db')
Session = sessionmaker(bind=engine)

# Create tables
def init_db():
    Base.metadata.create_all(engine)

# Utility function to create a session
def get_session():
    return Session()
