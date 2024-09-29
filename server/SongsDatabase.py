from sqlalchemy import engine, create_engine, select, Column, Integer, String, BLOB
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import pickle

# Define the base class
class Base(DeclarativeBase):
    pass

# The Song object that will be stored in the database
class Song(Base):
    __tablename__ = 'songs'  # Name of the table in the database

    song_id = Column(Integer, primary_key=True)  # Primary key
    track_id = Column(String(22))
    artists = Column(String(100))
    track_name = Column(String(100))
    album_name = Column(String(100))
    track_genre = Column(String(100))
    popularity = Column(Integer)
    #features will contain values used in the similarity score comparison
    #on the .csv table, they are the 6 values from energy inclusive to instrumentalness inclusive
    #they will be stored as a BLOB, so they are stored as a binary object
    features = Column(BLOB)

    def __repr__(self):
        return f"<Song(song id = {self.song_id}, track id = {self.track_id}, artists = {self.artists})>"
    
# Example setup to create the database
class SongsDatabase():
    def __init__(self):
        engine = create_engine('sqlite:///songs.db')
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)


    #adds a song to the database
    #features must be a list of floats that's 6 values long
    def addSong(self, song_id, track_id, artists, track_name, album_name, track_genre, popularity, features):
        with self.Session() as session:
            newSong = Song(song_id=song_id, track_id=track_id, artists=artists, track_name=track_name, album_name=album_name, track_genre=track_genre, popularity=popularity
, features=pickle.dumps(features))
            session.add(newSong)
            session.commit()
    
    def getSong(self, song_id):
        with self.Session() as session:
            statement = select(Song).filter_by(song_id=song_id)
            return session.scalars(statement).one_or_none()
    

    def deleteSong(self, song_id):
        with self.Session() as session:
            statement = select(Song).filter_by(song_id=song_id)
            targetSong = session.scalar(statement).one_or_none()
            if targetSong:
                session.delete(targetSong)
                session.commit()
            else:
                print(f"Song with id {song_id} not found.")


#used for testing
#songDatabase = SongsDatabase()
#songDatabase.addSong(0, "5SuOikwiRyPMVoIQDJUgSV", "Gen Hoshino", "Comedy", "Comedy", "acoustic", 73, [46.1, 3.2329317269076308,71.85929648241205,36.124533635751035,23.366013071895424,0.000101])
#print(songDatabase.getSong(1))