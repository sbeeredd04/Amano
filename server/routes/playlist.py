import logging
from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from models.song_model import Song, Playlist, PlaylistSong
from utils.db import get_session
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

playlists_bp = Blueprint('playlists', __name__)

# Configure CORS for the blueprint
CORS(playlists_bp, 
     origins=["http://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Fetch all playlists for a user
@playlists_bp.route('/get', methods=['GET', 'POST'])
def get_playlists():
    try:
        if request.method == 'POST':
            data = request.json
        else:
            data = request.args

        user_id = data.get('user_id')
        logger.debug(f"Fetching playlists for user_id: {user_id}")

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        session = get_session()
        try:
            playlists = session.query(Playlist).filter_by(user_id=user_id).all()
            if not playlists:
                logger.debug(f"No playlists found for user_id: {user_id}")
                return jsonify({"message": "No playlists found", "playlists": []}), 200

            playlists_with_songs = []
            for playlist in playlists:
                # Fetch songs for this playlist
                songs = session.query(Song)\
                    .join(PlaylistSong)\
                    .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                    .all()
                
                playlist_data = playlist.serialize()
                playlist_data['songs'] = [{
                    'song_id': song.song_id,
                    'track_name': song.track_name,
                    'artist_name': song.artists,
                    'track_genre': song.track_genre
                } for song in songs]
                
                playlists_with_songs.append(playlist_data)

            return jsonify({"playlists": playlists_with_songs}), 200

        except SQLAlchemyError as e:
            logger.error(f"Database error while fetching playlists: {str(e)}")
            return jsonify({"error": "Database error"}), 500
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Unexpected error in get_playlists: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Add a new playlist with selected songs
@playlists_bp.route('/add', methods=['POST'])
def add_playlist():
    logger.info("\n=== Adding New Playlist ===")
    
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    description = data.get('description', '')
    song_ids = data.get('song_ids', [])

    logger.info(f"Request Details:")
    logger.info(f"- User ID: {user_id}")
    logger.info(f"- Playlist Name: {name}")
    logger.info(f"- Description: {description}")
    logger.info(f"- Number of Songs: {len(song_ids)}")
    logger.info(f"- Song IDs: {song_ids}")

    session = get_session()
    try:
        # Log song details before adding
        songs_info = session.query(Song).filter(Song.song_id.in_(song_ids)).all()
        logger.info("\nSongs being added to playlist:")
        for song in songs_info:
            logger.info(f"- Song ID: {song.song_id}")
            logger.info(f"  Track: {song.track_name}")
            logger.info(f"  Artist: {song.artists}")
            logger.info(f"  Genre: {song.track_genre}")

        new_playlist = Playlist(user_id=user_id, name=name, description=description)
        session.add(new_playlist)
        session.commit()
        
        logger.info(f"\nCreated new playlist:")
        logger.info(f"- Playlist ID: {new_playlist.playlist_id}")
        logger.info(f"- Name: {new_playlist.name}")

        # Add songs to playlist
        for song_id in song_ids:
            playlist_song = PlaylistSong(playlist_id=new_playlist.playlist_id, song_id=song_id)
            session.add(playlist_song)
            logger.info(f"Added song {song_id} to playlist {new_playlist.playlist_id}")

        session.commit()
        logger.info(f"\nSuccessfully created playlist with {len(song_ids)} songs")
        
        return jsonify({
            "message": "Playlist added successfully", 
            "playlist": new_playlist.serialize()
        }), 201

    except Exception as e:
        session.rollback()
        logger.error(f"Error adding playlist: {str(e)}")
        logger.error(f"Error details:", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


# Fetch all songs with pagination, filtering, and search
@playlists_bp.route('/songs', methods=['GET'])
def get_songs():
    logger.info("\n=== Fetching Songs for Frontend ===")
    session = get_session()
    try:
        # Get and validate parameters
        genre = request.args.get('genre', '').strip()
        search = request.args.get('search', '').strip()
        search_type = request.args.get('type', 'track')
        
        logger.info(f"Query Parameters - Genre: {genre}, Search: {search}, Type: {search_type}")
        
        try:
            limit = int(request.args.get('limit', 20))
            offset = int(request.args.get('offset', 0))
        except (ValueError, TypeError):
            limit = 20
            offset = 0

        # Build query
        query = session.query(Song)
        logger.info("Initial song query built")

        # Apply filters and log
        if genre and genre.lower() != 'all':
            logger.info(f"Filtering by genre: {genre}")
            query = query.filter(Song.track_genre.ilike(f"%{genre}%"))

        if search:
            if search_type == 'artist':
                logger.info(f"Filtering by artist: {search}")
                query = query.filter(Song.artists.ilike(f"%{search}%"))
            else:
                logger.info(f"Filtering by track name: {search}")
                query = query.filter(Song.track_name.ilike(f"%{search}%"))
            
        # Get total count before pagination
        total_count = query.count()
        logger.info(f"Total matching songs: {total_count}")
        
        # Apply pagination
        songs = query.order_by(Song.popularity.desc()).offset(offset).limit(limit).all()
        
        # Log song details being sent to frontend
        logger.info("\nSongs being sent to frontend:")
        for song in songs:
            logger.info(f"- Song ID: {song.song_id}")
            logger.info(f"  Track: {song.track_name}")
            logger.info(f"  Artist: {song.artists}")
            logger.info(f"  Genre: {song.track_genre}")

        # Serialize results
        serialized_songs = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'album_name': song.album_name,
            'track_genre': song.track_genre,
            'popularity': song.popularity
        } for song in songs]

        logger.info(f"Successfully serialized {len(songs)} songs for frontend")
        return jsonify({
            "songs": serialized_songs,
            "total_count": total_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching songs: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "songs": [],
            "total_count": 0
        }), 500
    finally:
        session.close()


# Fetch all songs from the user's playlists
@playlists_bp.route('/user_songs', methods=['POST'])
def get_user_songs():
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    session = get_session()
    try:
        user_playlists = session.query(Playlist).filter_by(user_id=user_id).all()

        if not user_playlists:
            logger.debug(f"No playlists found for user_id {user_id}. Returning global song list.")
            return jsonify({"user_songs": [], "source": "global"}), 200

        user_song_ids = []
        for playlist in user_playlists:
            playlist_songs = session.query(PlaylistSong).filter_by(playlist_id=playlist.playlist_id).all()
            user_song_ids.extend([song.song_id for song in playlist_songs])

        if not user_song_ids:
            logger.debug(f"No songs found in playlists for user_id {user_id}. Returning global song list.")
            return jsonify({"user_songs": [], "source": "global"}), 200

        # Fetch detailed song info
        songs = session.query(Song).filter(Song.song_id.in_(user_song_ids)).all()
        songs_serialized = [
            {
                'song_id': song.song_id,
                'track_id': song.track_id,
                'track_name': song.track_name,
                'artist_name': song.artists,
                'album_name': song.album_name,
                'track_genre': song.track_genre,
                'popularity': song.popularity
            }
            for song in songs
        ]
        
        logger.debug(f"Fetched {len(songs_serialized)} user songs for user_id {user_id}.")
        return jsonify({"user_songs": songs_serialized, "source": "playlists"}), 200

    except Exception as e:
        logger.error(f"Error fetching user songs for user_id {user_id}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()


# Delete a playlist
@playlists_bp.route('/delete', methods=['POST'])
def delete_playlist():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')

    if not user_id or not playlist_id:
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        # First check if the playlist belongs to the user
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        # Delete associated playlist songs first
        session.query(PlaylistSong).filter_by(playlist_id=playlist_id).delete()
        # Delete the playlist
        session.delete(playlist)
        session.commit()

        return jsonify({"message": "Playlist deleted successfully"}), 200

    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting playlist: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()


# Edit a playlist
@playlists_bp.route('/edit', methods=['POST'])
def edit_playlist():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')
    name = data.get('name')
    song_ids = data.get('song_ids', [])

    if not all([user_id, playlist_id, name]):
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        # Check if playlist belongs to user
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        # Update playlist details
        playlist.name = name

        # Update songs
        # First, remove all existing songs
        session.query(PlaylistSong).filter_by(playlist_id=playlist_id).delete()
        
        # Add new songs
        for song_id in song_ids:
            playlist_song = PlaylistSong(playlist_id=playlist_id, song_id=song_id)
            session.add(playlist_song)

        session.commit()

        # Fetch updated playlist with songs for response
        updated_songs = session.query(Song)\
            .join(PlaylistSong)\
            .filter(PlaylistSong.playlist_id == playlist_id)\
            .all()

        response_data = playlist.serialize()
        response_data['songs'] = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'track_genre': song.track_genre
        } for song in updated_songs]

        return jsonify({
            "message": "Playlist updated successfully",
            "playlist": response_data
        }), 200

    except Exception as e:
        session.rollback()
        logger.error(f"Error updating playlist: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()


@playlists_bp.route('/genres', methods=['GET'])
def get_genres():
    session = get_session()
    try:
        # Get unique genres and add 'All' option
        genres = ['All'] + [g[0] for g in session.query(Song.track_genre.distinct()).all()]
        return jsonify({"genres": genres}), 200
    except Exception as e:
        logger.error(f"Error fetching genres: {e}")
        return jsonify({"error": str(e)}), 500

@playlists_bp.route('/artists', methods=['GET'])
def get_artists():
    session = get_session()
    try:
        # Get unique artists
        artists = [a[0] for a in session.query(Song.artists.distinct()).all()]
        return jsonify({"artists": artists}), 200
    except Exception as e:
        logger.error(f"Error fetching artists: {e}")
        return jsonify({"error": str(e)}), 500

@playlists_bp.route('/remove_song', methods=['POST'])
def remove_song():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')
    song_id = data.get('song_id')

    if not all([user_id, playlist_id, song_id]):
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        # Check if playlist belongs to user
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        # Remove the song from the playlist
        session.query(PlaylistSong).filter_by(
            playlist_id=playlist_id,
            song_id=song_id
        ).delete()

        session.commit()

        # Fetch updated playlist with remaining songs
        updated_songs = session.query(Song)\
            .join(PlaylistSong)\
            .filter(PlaylistSong.playlist_id == playlist_id)\
            .all()

        response_data = playlist.serialize()
        response_data['songs'] = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'track_genre': song.track_genre
        } for song in updated_songs]

        return jsonify({
            "message": "Song removed successfully",
            "playlist": response_data
        }), 200

    except Exception as e:
        session.rollback()
        logger.error(f"Error removing song from playlist: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()

def get_all_user_playlist_songs(user_id):
    """
    Get all unique songs from all playlists of a user.
    """
    logger.info(f"Fetching all songs from all playlists for user {user_id}")
    
    session = get_session()
    try:
        # Get all playlists for the user
        playlists = session.query(Playlist).filter_by(user_id=user_id).all()
        logger.info(f"Found {len(playlists)} playlists")
        
        # Get all unique songs from all playlists
        all_songs = set()
        for playlist in playlists:
            songs = session.query(Song)\
                .join(PlaylistSong)\
                .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                .all()
            all_songs.update(song.song_id for song in songs)
        
        logger.info(f"Total unique songs across all playlists: {len(all_songs)}")
        return list(all_songs)
        
    except Exception as e:
        logger.error(f"Error fetching user playlist songs: {e}")
        return []
    finally:
        session.close()
