# server/routes/playlists.py
from flask import Blueprint, request, jsonify
from services.spotify_service import get_playlist_tracks
from models.song_model import Song
from utils.db import get_session
import logging

playlists_bp = Blueprint('playlists', __name__)

@playlists_bp.route('/', methods=['POST'])
def handle_playlists():
    
    #debugging
    print("playlists_bp.route")
    
    logging.basicConfig(level=logging.DEBUG)

    data = request.json
    logging.debug(f"Received data: {data}")
    
    playlist_ids = data.get('playlistIds')
    token = data.get('token')
    
    logging.debug(f"Playlist IDs: {playlist_ids}")
    logging.debug(f"Token: {token}")

    all_tracks = []
    
    # Open a database session
    session = get_session()
    
    try:
        for playlist_id in playlist_ids:
            # Fetch the tracks from Spotify API using the token
            tracks = get_playlist_tracks(playlist_id, token)
            all_tracks.extend(tracks)
            
            # Save tracks to the database
            for track in tracks:
                # Check if the song already exists in the database
                existing_song = session.query(Song).filter_by(track_id=track['id']).first()
                
                if not existing_song:
                    new_song = Song(
                        track_id=track['id'],
                        artists=track['artist'],
                        track_name=track['name'],
                        album_name=track['album'],
                        track_genre="Unknown",
                        popularity=0
                    )
                    session.add(new_song)
                    
        # Commit the new songs to the database
        session.commit()

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()

    return jsonify({"tracks": all_tracks, "track_count": len(all_tracks)})
